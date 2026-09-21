//
// Created by Göksu Güvendiren on 2019-05-14.
//

#include "Scene.hpp"

void Scene::buildBVH() {
  printf(" - Generating BVH...\n\n");
  this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);
}

Intersection Scene::intersect(const Ray &ray) const {
  return this->bvh->Intersect(ray);
}

void Scene::sampleLight(Intersection &pos, float &pdf) const {
  float emit_area_sum = 0;
  for (uint32_t k = 0; k < objects.size(); ++k) {
    if (objects[k]->hasEmit()) {
      emit_area_sum += objects[k]->getArea();
    }
  }
  float p = get_random_float() * emit_area_sum;
  emit_area_sum = 0;
  for (uint32_t k = 0; k < objects.size(); ++k) {
    if (objects[k]->hasEmit()) {
      emit_area_sum += objects[k]->getArea();
      if (p <= emit_area_sum) {
        objects[k]->Sample(pos, pdf);
        break;
      }
    }
  }
}

bool Scene::trace(const Ray &ray, const std::vector<Object *> &objects,
                  float &tNear, uint32_t &index, Object **hitObject) {
  *hitObject = nullptr;
  for (uint32_t k = 0; k < objects.size(); ++k) {
    float tNearK = kInfinity;
    uint32_t indexK;
    Vector2f uvK;
    if (objects[k]->intersect(ray, tNearK, indexK) && tNearK < tNear) {
      *hitObject = objects[k];
      tNear = tNearK;
      index = indexK;
    }
  }

  return (*hitObject != nullptr);
}

// Implementation of Path Tracing
Vector3f Scene::castRay(const Ray &ray, int depth) const {
  Intersection inter = intersect(ray);
  if (!inter.happened) {
    return Vector3f(0.0f); // Cornell box 一般返回黑色背景
  }

  // 射线直接打到光源
  if (inter.m->hasEmission()) {
    return inter.m->getEmission();
  }

  Vector3f p = inter.coords;
  Vector3f N = inter.normal.normalized();
  Vector3f wo = -ray.direction; // 出射方向（从交点指向相机）

  // ---------- 1) 直接光照 L_dir ----------
  Vector3f L_dir(0.0f);
  Intersection lightInter;
  float pdf_light = 0.0f;
  sampleLight(lightInter, pdf_light);

  Vector3f x = lightInter.coords;
  Vector3f NN = lightInter.normal.normalized();
  Vector3f ws = (x - p).normalized(); // 指向光源采样点
  float dist2 = dotProduct(x - p, x - p);

  Ray shadowRay(p + N * EPSILON, ws);
  Intersection occ = intersect(shadowRay);

  // 无遮挡：第一个命中点就是光源采样点附近
  float dist = std::sqrt(dist2);
  if (occ.happened && std::fabs(occ.distance - dist) < 1e-2f) {
    float cos1 = std::max(0.0f, dotProduct(ws, N));
    float cos2 = std::max(0.0f, dotProduct(-ws, NN));
    Vector3f fr = inter.m->eval(wo, ws, N);

    L_dir = lightInter.emit * fr * cos1 * cos2 / dist2 / pdf_light;
  }

  // ---------- 2) 间接光照 L_indir ----------
  Vector3f L_indir(0.0f);
  if (get_random_float() < RussianRoulette) {
    Vector3f wi = inter.m->sample(wo, N).normalized();
    float pdf = inter.m->pdf(wo, wi, N);

    if (pdf > EPSILON) {
      Ray newRay(p + N * EPSILON, wi);
      Intersection nextInter = intersect(newRay);

      // 命中非发光体才走递归，避免与直接光重复计数
      if (nextInter.happened && !nextInter.m->hasEmission()) {
        float cosTheta = std::max(0.0f, dotProduct(wi, N));
        Vector3f fr = inter.m->eval(wo, wi, N);

        L_indir =
            castRay(newRay, depth + 1) * fr * cosTheta / pdf / RussianRoulette;
      }
    }
  }

  return L_dir + L_indir;
}