#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "OBJ_Loader.h"
#include "Shader.hpp"
#include "Texture.hpp"
#include "Triangle.hpp"
#include "global.hpp"
#include "rasterizer.hpp"

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos) {
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0],
        0, 1, 0, -eye_pos[1],
        0, 0, 1, -eye_pos[2],
        0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float angle) {
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
        0, 1, 0, 0,
        -sin(angle), 0, cos(angle), 0,
        0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
        0, 2.5, 0, 0,
        0, 0, 2.5, 0,
        0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    return translate * rotation * scale;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar) {
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
    projection(0, 0) = aspect_ratio / std::tan(eye_fov / 2);
    projection(1, 1) = 1 / std::tan(eye_fov / 2);
    projection(2, 2) = (zNear + zFar) / (zNear - zFar);
    projection(2, 3) = 2 * zNear * zFar / (zNear - zFar);
    projection(3, 2) = -1;
    projection(3, 3) = 0;
    return projection;
}

Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload) {
    return payload.position;
}

Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload) {
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f& vec, const Eigen::Vector3f& axis) {
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

struct light {
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload) {
    Eigen::Vector3f return_color = {0, 0, 0};
    if (payload.texture && payload.texture->width > 0 && payload.texture->height > 0) {
        // Get the texture value at the texture coordinates of the current fragment
        // ----------- 1. 纹理采样 -----------
        auto uv = payload.tex_coords;

        constexpr float kUvMax = 0.999999f;
        float u = std::isfinite(uv.x()) ? std::clamp(uv.x(), 0.0f, kUvMax) : 0.5f;
        float v = std::isfinite(uv.y()) ? std::clamp(uv.y(), 0.0f, kUvMax) : 0.5f;

        return_color = payload.texture->getColor(u, v);
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights) {
        // for each light source in the code, calculate what the *ambient*, *diffuse*, and *specular*
        // ----------- 光照方向 -----------
        Eigen::Vector3f light_dir = (light.position - point).normalized();
        Eigen::Vector3f view_dir = (eye_pos - point).normalized();

        float r2 = (light.position - point).squaredNorm();

        // ----------- Ambient -----------
        Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);

        // ----------- Diffuse -----------
        float cos_theta = std::max(0.0f, normal.dot(light_dir));
        Eigen::Vector3f diffuse =
            kd.cwiseProduct(light.intensity / r2) * cos_theta;

        // ----------- Specular (Blinn-Phong) -----------
        Eigen::Vector3f h = (light_dir + view_dir).normalized();
        float cos_alpha = std::max(0.0f, normal.dot(h));

        Eigen::Vector3f specular =
            ks.cwiseProduct(light.intensity / r2) * std::pow(cos_alpha, p);

        // ----------- 累加 -----------
        result_color += ambient + diffuse + specular;

        // components are. Then, accumulate that result on the *result_color* object.
    }

    return result_color * 255.f;
}

Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload) {
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};
    for (auto& light : lights) {
        // For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular*
        //  components are. Then, accumulate that result on the *result_color* object.
        //  光源方向
        Eigen::Vector3f light_dir = (light.position - point).normalized();

        // 视线方向（相机在 view space 是原点）
        Eigen::Vector3f view_dir = (eye_pos - point).normalized();

        // 距离平方（用于衰减）
        float r2 = (light.position - point).squaredNorm();
        // ----------- Ambient（环境光）-----------
        Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);
        // ----------- Diffuse（漫反射）-----------
        float cos_theta = std::max(0.0f, normal.normalized().dot(light_dir));
        Eigen::Vector3f diffuse =
            kd.cwiseProduct(light.intensity / r2) * cos_theta;
        // ----------- Specular（高光，Blinn-Phong）-----------
        Eigen::Vector3f h = (light_dir + view_dir).normalized();
        float cos_alpha = std::max(0.0f, normal.normalized().dot(h));
        Eigen::Vector3f specular =
            ks.cwiseProduct(light.intensity / r2) * std::pow(cos_alpha, p);
        // ----------- 累加 -----------
        result_color += ambient + diffuse + specular;
    }

    return result_color * 255.f;
}

Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload& payload) {

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal.normalized();

    float kh = 0.2, kn = 0.1;
    if (!payload.texture)
        return Eigen::Vector3f(0, 0, 0);

    // Implement displacement mapping here
    // ----------- 1. 构造 TBN -----------
    float x = normal.x();
    float y = normal.y();
    float z = normal.z();

    Eigen::Vector3f t = Eigen::Vector3f(
        x * y / std::sqrt(x * x + z * z),
        std::sqrt(x * x + z * z),
        z * y / std::sqrt(x * x + z * z));

    Eigen::Vector3f b = normal.cross(t);

    Eigen::Matrix3f TBN;
    TBN << t, b, normal;

    // ----------- 2. 计算高度梯度 -----------
    auto uv = payload.tex_coords;

    constexpr float kUvMax = 0.999999f;
    float u = std::clamp(uv.x(), 0.0f, kUvMax);
    float v = std::clamp(uv.y(), 0.0f, kUvMax);

    float w = payload.texture->width;
    float h = payload.texture->height;
    if (w <= 0 || h <= 0) {
        return Eigen::Vector3f(0, 0, 0);
    }

    float huv = payload.texture->getColor(u, v).norm();
    float u1 = std::clamp(u + 1.0f / w, 0.0f, kUvMax);
    float v1 = std::clamp(v + 1.0f / h, 0.0f, kUvMax);

    float huv_u = payload.texture->getColor(u1, v).norm();
    float huv_v = payload.texture->getColor(u, v1).norm();

    float dU = kh * kn * (huv_u - huv);
    float dV = kh * kn * (huv_v - huv);

    // ----------- 3. 更新位置（关键区别！！）-----------
    point = point + kn * normal * huv;

    // ----------- 4. 更新法线 -----------
    Eigen::Vector3f ln(-dU, -dV, 1.0f);
    normal = (TBN * ln).normalized();

    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Position p = p + kn * n * h(u,v)
    // Normal n = normalize(TBN * ln)

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights) {
        //  For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular*
        Eigen::Vector3f light_dir = (light.position - point).normalized();
        Eigen::Vector3f view_dir = (eye_pos - point).normalized();

        float r2 = (light.position - point).squaredNorm();

        // Ambient
        Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);

        // Diffuse
        float cos_theta = std::max(0.0f, normal.dot(light_dir));
        Eigen::Vector3f diffuse =
            kd.cwiseProduct(light.intensity / r2) * cos_theta;

        // Specular
        Eigen::Vector3f h_vec = (light_dir + view_dir).normalized();
        float cos_alpha = std::max(0.0f, normal.dot(h_vec));
        Eigen::Vector3f specular =
            ks.cwiseProduct(light.intensity / r2) * std::pow(cos_alpha, p);

        result_color += ambient + diffuse + specular;
        // components are. Then, accumulate that result on the *result_color* object.
    }

    return result_color * 255.f;
}

Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload) {


    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal.normalized();

    float kh = 0.2, kn = 0.1;
    // ----------- 1. 构造 TBN 矩阵 -----------
    float x = normal.x();
    float y = normal.y();
    float z = normal.z();

    Eigen::Vector3f t = Eigen::Vector3f(
        x * y / std::sqrt(x * x + z * z),
        std::sqrt(x * x + z * z),
        z * y / std::sqrt(x * x + z * z));

    Eigen::Vector3f b = normal.cross(t);

    Eigen::Matrix3f TBN;
    TBN << t, b, normal;

    // ----------- 2. 计算 dU / dV（高度变化）-----------
    auto uv = payload.tex_coords;

    constexpr float kUvMax = 0.999999f;
    float u = std::clamp(uv.x(), 0.0f, kUvMax);
    float v = std::clamp(uv.y(), 0.0f, kUvMax);


    float w = payload.texture->width;
    float h = payload.texture->height;
    if (w <= 0 || h <= 0) {
        return Eigen::Vector3f(0, 0, 0);
    }

    // h(u,v)
    float huv = payload.texture->getColor(u, v).norm();

    float u1 = std::clamp(u + 1.0f / w, 0.0f, kUvMax);
    float v1 = std::clamp(v + 1.0f / h, 0.0f, kUvMax);

    float huv_u = payload.texture->getColor(u1, v).norm();
    float huv_v = payload.texture->getColor(u, v1).norm();

    float dU = kh * kn * (huv_u - huv);
    float dV = kh * kn * (huv_v - huv);
    // ----------- 3. 局部法线 ln -----------
    Eigen::Vector3f ln(-dU, -dV, 1.0f);
    // ----------- 4. 转到世界 / view space -----------
    Eigen::Vector3f new_normal = (TBN * ln).normalized();

    // Implement bump mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Normal n = normalize(TBN * ln)

    Eigen::Vector3f result_color = (new_normal + Eigen::Vector3f(1, 1, 1)) / 2.0f;

    return result_color * 255.f;
}

int main(int argc, const char** argv) {
    std::vector<Triangle*> TriangleList;

    float angle = 140.0;
    bool command_line = false;
    int ssaa_scale = 1;

    std::string filename = "output.png";
    objl::Loader Loader;
    std::string obj_path = "../models/spot/";

    // Load .obj File
    bool loadout = Loader.LoadFile("../models/spot/spot_triangulated_good.obj");
    for (auto mesh : Loader.LoadedMeshes) {
        for (int i = 0; i < mesh.Vertices.size(); i += 3) {
            Triangle* t = new Triangle();
            for (int j = 0; j < 3; j++) {
                t->setVertex(j, Vector4f(mesh.Vertices[i + j].Position.X, mesh.Vertices[i + j].Position.Y, mesh.Vertices[i + j].Position.Z, 1.0));
                t->setNormal(j, Vector3f(mesh.Vertices[i + j].Normal.X, mesh.Vertices[i + j].Normal.Y, mesh.Vertices[i + j].Normal.Z));
                t->setTexCoord(j, Vector2f(mesh.Vertices[i + j].TextureCoordinate.X, mesh.Vertices[i + j].TextureCoordinate.Y));
            }
            TriangleList.push_back(t);
        }
    }

    auto texture_path = "hmap.jpg";

    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = phong_fragment_shader;

    if (argc >= 2) {
        command_line = true;
        filename = std::string(argv[1]);

        if (argc >= 3) {
            std::string mode = argv[2];
            if (mode == "texture") {
                std::cout << "Rasterizing using the texture shader\n";
                active_shader = texture_fragment_shader;
                texture_path = "spot_texture.png";
            } else if (mode == "normal") {
                std::cout << "Rasterizing using the normal shader\n";
                active_shader = normal_fragment_shader;
            } else if (mode == "phong") {
                std::cout << "Rasterizing using the phong shader\n";
                active_shader = phong_fragment_shader;
            } else if (mode == "bump") {
                std::cout << "Rasterizing using the bump shader\n";
                active_shader = bump_fragment_shader;
            } else if (mode == "displacement") {
                std::cout << "Rasterizing using the displacement shader\n";
                active_shader = displacement_fragment_shader;
            }
        }

        if (argc >= 4) {
            std::string sr_arg = argv[3];
            if (sr_arg == "sr2") ssaa_scale = 2;
            else if (sr_arg == "sr3") ssaa_scale = 3;
            else if (sr_arg == "sr4") ssaa_scale = 4;
            if (ssaa_scale > 1) {
                std::cout << "Super-resolution enabled: " << ssaa_scale << "x\n";
            }
        }
    }

    rst::rasterizer r(700 * ssaa_scale, 700 * ssaa_scale);
    r.set_texture(Texture(obj_path + texture_path));

    Eigen::Vector3f eye_pos = {0, 0, 10};

    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        r.draw(TriangleList);
        cv::Mat render_image(700 * ssaa_scale, 700 * ssaa_scale, CV_32FC3, r.frame_buffer().data());
        cv::Mat image;
        render_image.convertTo(image, CV_8UC3, 1.0f);
        if (ssaa_scale > 1) {
            cv::resize(image, image, cv::Size(700, 700), 0, 0, cv::INTER_AREA);
        }
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        // r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        r.draw(TriangleList);
        cv::Mat render_image(700 * ssaa_scale, 700 * ssaa_scale, CV_32FC3, r.frame_buffer().data());
        cv::Mat image;
        render_image.convertTo(image, CV_8UC3, 1.0f);
        if (ssaa_scale > 1) {
            cv::resize(image, image, cv::Size(700, 700), 0, 0, cv::INTER_AREA);
        }
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);

        if (key == 'a') {
            angle -= 0.1;
        } else if (key == 'd') {
            angle += 0.1;
        }
    }
    return 0;
}
