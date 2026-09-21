//
// Created by goksu on 2/25/20.
//

#include "Renderer.hpp"
#include "Scene.hpp"
#include <atomic>
#include <chrono>
#include <thread>

inline float deg2rad(const float &deg) { return deg * M_PI / 180.0; }

const float EPSILON = 0.00001;

// The main render function. This where we iterate over all pixels in the image,
// generate primary rays and cast these rays into the scene. The content of the
// framebuffer is saved to a file.
void Renderer::Render(const Scene &scene) {
  std::vector<Vector3f> framebuffer(scene.width * scene.height);

  float scale = tan(deg2rad(scene.fov * 0.5));
  float imageAspectRatio = scene.width / (float)scene.height;
  Vector3f eye_pos(278, 273, -800);

  // change the spp value to change sample ammount
  int spp = 128;
  std::cout << "SPP: " << spp << "\n";
  const int width = scene.width;
  const int height = scene.height;
  unsigned int threadCount = std::max(1u, std::thread::hardware_concurrency());
  std::cout << "Threads: " << threadCount << "\n";

  std::atomic<int> finishedRows(0);
  std::vector<std::thread> workers;
  workers.reserve(threadCount);

  auto renderRows = [&](int rowBegin, int rowEnd) {
    for (int j = rowBegin; j < rowEnd; ++j) {
      for (int i = 0; i < width; ++i) {
        float x =
            (2 * (i + 0.5f) / (float)width - 1.f) * imageAspectRatio * scale;
        float y = (1.f - 2 * (j + 0.5f) / (float)height) * scale;

        Vector3f dir = normalize(Vector3f(-x, y, 1));
        Vector3f color(0);
        for (int k = 0; k < spp; ++k) {
          color += scene.castRay(Ray(eye_pos, dir), 0) / spp;
        }
        framebuffer[j * width + i] = color;
      }
      ++finishedRows;
    }
  };

  int rowsPerThread = (height + (int)threadCount - 1) / (int)threadCount;
  for (unsigned int t = 0; t < threadCount; ++t) {
    int rowBegin = (int)t * rowsPerThread;
    int rowEnd = std::min(height, rowBegin + rowsPerThread);
    if (rowBegin < rowEnd) {
      workers.emplace_back(renderRows, rowBegin, rowEnd);
    }
  }

  while (finishedRows < height) {
    UpdateProgress(finishedRows.load() / (float)height);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  for (auto &worker : workers) {
    worker.join();
  }
  UpdateProgress(1.f);

  // save framebuffer to file
  FILE *fp = fopen("binary.ppm", "wb");
  (void)fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);
  for (auto i = 0; i < scene.height * scene.width; ++i) {
    static unsigned char color[3];
    color[0] =
        (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].x), 0.6f));
    color[1] =
        (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].y), 0.6f));
    color[2] =
        (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].z), 0.6f));
    fwrite(color, 1, 3, fp);
  }
  fclose(fp);
}
