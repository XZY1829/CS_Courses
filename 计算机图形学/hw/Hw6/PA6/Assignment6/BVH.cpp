#include "BVH.hpp"
#include <algorithm>
#include <cassert>

BVHAccel::BVHAccel(std::vector<Object *> p, int maxPrimsInNode,
                   SplitMethod splitMethod)
    : maxPrimsInNode(std::min(255, maxPrimsInNode)), splitMethod(splitMethod),
      primitives(std::move(p)) {
  time_t start, stop;
  time(&start);
  if (primitives.empty())
    return;

  root = recursiveBuild(primitives);

  time(&stop);
  double diff = difftime(stop, start);
  int hrs = (int)diff / 3600;
  int mins = ((int)diff / 60) - (hrs * 60);
  int secs = (int)diff - (hrs * 3600) - (mins * 60);

  printf(
      "\rBVH Generation complete: \nTime Taken: %i hrs, %i mins, %i secs\n\n",
      hrs, mins, secs);
}

BVHBuildNode *BVHAccel::recursiveBuild(std::vector<Object *> objects) {
  BVHBuildNode *node = new BVHBuildNode();

  // Compute bounds of all primitives in BVH node
  Bounds3 bounds;
  for (int i = 0; i < objects.size(); ++i)
    bounds = Union(bounds, objects[i]->getBounds());
  if (objects.size() == 1) {
    // Create leaf _BVHBuildNode_
    node->bounds = objects[0]->getBounds();
    node->object = objects[0];
    node->left = nullptr;
    node->right = nullptr;
    return node;
  } else if (objects.size() == 2) {
    node->left = recursiveBuild(std::vector{objects[0]});
    node->right = recursiveBuild(std::vector{objects[1]});

    node->bounds = Union(node->left->bounds, node->right->bounds);
    return node;
  } else {
    Bounds3 centroidBounds;
    for (int i = 0; i < objects.size(); ++i)
      centroidBounds =
          Union(centroidBounds, objects[i]->getBounds().Centroid());
    int dim = centroidBounds.maxExtent();
    switch (dim) {
    case 0:
      std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
        return f1->getBounds().Centroid().x < f2->getBounds().Centroid().x;
      });
      break;
    case 1:
      std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
        return f1->getBounds().Centroid().y < f2->getBounds().Centroid().y;
      });
      break;
    case 2:
      std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
        return f1->getBounds().Centroid().z < f2->getBounds().Centroid().z;
      });
      break;
    }

    auto beginning = objects.begin();
    auto middling = objects.begin() + (objects.size() / 2);
    auto ending = objects.end();

    auto leftshapes = std::vector<Object *>(beginning, middling);
    auto rightshapes = std::vector<Object *>(middling, ending);

    assert(objects.size() == (leftshapes.size() + rightshapes.size()));

    node->left = recursiveBuild(leftshapes);
    node->right = recursiveBuild(rightshapes);

    node->bounds = Union(node->left->bounds, node->right->bounds);
  }

  return node;
}

Intersection BVHAccel::Intersect(const Ray &ray) const {
  Intersection isect;
  if (!root)
    return isect;
  isect = BVHAccel::getIntersection(root, ray);
  return isect;
}

// BVH遍历的基本思路：
// 1. 判断当前node的包围盒是否和ray相交，如果不相交直接返回空Intersection。
// 2.
// 如果是叶子节点，则对叶子节点包含的具体primitive（如三角形等）进行相交测试，并返回相交信息。
// 3. 如果不是叶子节点，则递归查找左右子节点，找到距离最近的Intersection返回。

Intersection BVHAccel::getIntersection(BVHBuildNode *node,
                                       const Ray &ray) const {
  Intersection inter, inter_left, inter_right;

  // 1. 判断包围盒是否和光线相交，不相交直接返回
  if (!node->bounds.IntersectP(ray, ray.direction_inv, {0, 0, 0})) {
    return inter; // 没有发生碰撞，inter.happened=false
  }

  // 2. 叶节点（说明node里存了primitive，通常只存一个）
  if (node->left == nullptr && node->right == nullptr) {
    // node->object 指向叶子primitive，例如一个Triangle
    if (node->object)
      inter = node->object->getIntersection(ray);
    return inter;
  }

  // 3. 非叶子节点，递归左右子树
  inter_left = getIntersection(node->left, ray);
  inter_right = getIntersection(node->right, ray);

  // 4. 取离ray.origin最近的那个相交点（如果发生了相交）
  if (inter_left.happened && inter_right.happened) {
    if (inter_left.distance < inter_right.distance)
      return inter_left;
    else
      return inter_right;
  } else if (inter_left.happened) {
    return inter_left;
  } else if (inter_right.happened) {
    return inter_right;
  }
  // 都没命中
  return inter;
}