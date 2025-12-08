#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>

// ==================== 量化位宽配置 ====================

// 量化位宽枚举
// 注意：使用负值表示无符号类型，便于区分
enum class QuantBitWidth : int8_t {
    INT8 = 8,
    INT16 = 16,
    INT32 = 32,   // 用于中间累加
    UINT8 = -8,   // 无符号 8 位
    UINT16 = -16  // 无符号 16 位
};

// ==================== 编译期类型选择器 ====================
// 根据位宽选择量化类型（编译期）
template<QuantBitWidth BW>
struct QuantTypeSelector;

template<>
struct QuantTypeSelector<QuantBitWidth::INT8> {
    using type = int8_t;
    static constexpr int32_t min_val = -128;
    static constexpr int32_t max_val = 127;
    static constexpr bool is_signed = true;
    static constexpr int bits = 8;
};

template<>
struct QuantTypeSelector<QuantBitWidth::INT16> {
    using type = int16_t;
    static constexpr int32_t min_val = -32768;
    static constexpr int32_t max_val = 32767;
    static constexpr bool is_signed = true;
    static constexpr int bits = 16;
};

template<>
struct QuantTypeSelector<QuantBitWidth::INT32> {
    using type = int32_t;
    static constexpr int64_t min_val = INT32_MIN;
    static constexpr int64_t max_val = INT32_MAX;
    static constexpr bool is_signed = true;
    static constexpr int bits = 32;
};

template<>
struct QuantTypeSelector<QuantBitWidth::UINT8> {
    using type = uint8_t;
    static constexpr int32_t min_val = 0;
    static constexpr int32_t max_val = 255;
    static constexpr bool is_signed = false;
    static constexpr int bits = 8;
};

template<>
struct QuantTypeSelector<QuantBitWidth::UINT16> {
    using type = uint16_t;
    static constexpr int32_t min_val = 0;
    static constexpr int32_t max_val = 65535;
    static constexpr bool is_signed = false;
    static constexpr int bits = 16;
};

// 类型别名：根据位宽获取对应的量化类型
template<QuantBitWidth BW>
using QuantType = typename QuantTypeSelector<BW>::type;

// ==================== 运行时位宽信息 ====================
// 用于在运行时获取位宽相关信息
struct RuntimeBitWidthInfo {
    int32_t min_val;
    int32_t max_val;
    bool is_signed;
    int bits;
    size_t byte_size;

    // 根据 QuantBitWidth 枚举获取运行时信息
    static RuntimeBitWidthInfo fromBitWidth(QuantBitWidth bw) {
        RuntimeBitWidthInfo info;
        switch (bw) {
            case QuantBitWidth::INT8:
                info = {-128, 127, true, 8, sizeof(int8_t)};
                break;
            case QuantBitWidth::INT16:
                info = {-32768, 32767, true, 16, sizeof(int16_t)};
                break;
            case QuantBitWidth::INT32:
                info = {INT32_MIN, INT32_MAX, true, 32, sizeof(int32_t)};
                break;
            case QuantBitWidth::UINT8:
                info = {0, 255, false, 8, sizeof(uint8_t)};
                break;
            case QuantBitWidth::UINT16:
                info = {0, 65535, false, 16, sizeof(uint16_t)};
                break;
            default:
                throw std::invalid_argument("Unknown QuantBitWidth");
        }
        return info;
    }

    // 从整数值获取运行时信息
    static RuntimeBitWidthInfo fromInt(int8_t bw_int) {
        return fromBitWidth(static_cast<QuantBitWidth>(bw_int));
    }
};

// ==================== 运行时位宽分发器 ====================
// 核心：根据运行时位宽值调用正确的模板实例

/**
 * @brief 通用位宽分发器
 * @tparam Func 可调用对象类型，签名为 template<typename QuantT> ReturnType(Args...)
 * @param bw 运行时位宽枚举
 * @param func 要调用的模板函数包装器
 *
 * 使用示例：
 * ```cpp
 * auto result = dispatchByBitWidth(QuantBitWidth::INT8, [&](auto type_tag) {
 *     using QuantT = typename decltype(type_tag)::type;
 *     return someFunction<QuantT>(args...);
 * });
 * ```
 */

// 类型标签，用于传递类型信息
template<typename T>
struct TypeTag {
    using type = T;
};

// 分发器实现
template<typename Func>
auto dispatchByBitWidth(QuantBitWidth bw, Func&& func)
    -> decltype(func(TypeTag<int8_t>{})) {
    switch (bw) {
        case QuantBitWidth::INT8:
            return func(TypeTag<int8_t>{});
        case QuantBitWidth::INT16:
            return func(TypeTag<int16_t>{});
        case QuantBitWidth::INT32:
            return func(TypeTag<int32_t>{});
        case QuantBitWidth::UINT8:
            return func(TypeTag<uint8_t>{});
        case QuantBitWidth::UINT16:
            return func(TypeTag<uint16_t>{});
        default:
            throw std::invalid_argument("Unknown QuantBitWidth: " +
                                        std::to_string(static_cast<int>(bw)));
    }
}

// 从整数值分发（Python绑定常用）
template<typename Func>
auto dispatchByBitWidthInt(int8_t bw_int, Func&& func)
    -> decltype(func(TypeTag<int8_t>{})) {
    return dispatchByBitWidth(static_cast<QuantBitWidth>(bw_int),
                              std::forward<Func>(func));
}

// ==================== 双位宽分发器 ====================
// 用于需要同时处理两种不同位宽的情况（如输入和输出位宽不同）

template<typename Func>
auto dispatchByBitWidthPair(QuantBitWidth bw1, QuantBitWidth bw2, Func&& func)
    -> decltype(func(TypeTag<int8_t>{}, TypeTag<int8_t>{})) {
    return dispatchByBitWidth(bw1, [&](auto tag1) {
        return dispatchByBitWidth(bw2, [&](auto tag2) {
            return func(tag1, tag2);
        });
    });
}

// ==================== 算子量化位宽配置结构 ====================
struct OperatorQuantConfig {
    QuantBitWidth x_bitwidth = QuantBitWidth::INT8;           // 输入 x
    QuantBitWidth h_bitwidth = QuantBitWidth::INT8;           // 隐藏状态 h
    QuantBitWidth W_bitwidth = QuantBitWidth::INT8;           // 权重 W
    QuantBitWidth R_bitwidth = QuantBitWidth::INT8;           // 权重 R
    QuantBitWidth bx_bitwidth = QuantBitWidth::INT32;         // 偏置 bx (通常用 int32)
    QuantBitWidth br_bitwidth = QuantBitWidth::INT32;         // 偏置 br (通常用 int32)
    QuantBitWidth Wx_bitwidth = QuantBitWidth::INT32;         // Wx 中间结果
    QuantBitWidth Rh_bitwidth = QuantBitWidth::INT32;         // Rh 中间结果
    QuantBitWidth z_pre_bitwidth = QuantBitWidth::INT8;       // z 门输入
    QuantBitWidth z_out_bitwidth = QuantBitWidth::INT8;       // z 门输出
    QuantBitWidth r_pre_bitwidth = QuantBitWidth::INT8;       // r 门输入
    QuantBitWidth r_out_bitwidth = QuantBitWidth::INT8;       // r 门输出
    QuantBitWidth g_pre_bitwidth = QuantBitWidth::INT8;       // g 门输入
    QuantBitWidth g_out_bitwidth = QuantBitWidth::INT8;       // g 门输出
    QuantBitWidth Rh_add_br_bitwidth = QuantBitWidth::INT32;  // Rh + br
    QuantBitWidth rRh_bitwidth = QuantBitWidth::INT32;        // r × Rh
    QuantBitWidth one_minus_update_bitwidth = QuantBitWidth::INT8;
    QuantBitWidth old_contrib_bitwidth = QuantBitWidth::INT32;
    QuantBitWidth new_contrib_bitwidth = QuantBitWidth::INT32;

    // 默认构造函数：使用 int8 量化
    OperatorQuantConfig() = default;

    // 从整数配置创建（用于 Python 绑定）
    static OperatorQuantConfig fromInts(
        int8_t x = 8, int8_t h = 8, int8_t W = 8, int8_t R = 8,
        int8_t bx = 32, int8_t br = 32,
        int8_t Wx = 32, int8_t Rh = 32,
        int8_t z_pre = 8, int8_t z_out = 8,
        int8_t r_pre = 8, int8_t r_out = 8,
        int8_t g_pre = 8, int8_t g_out = 8,
        int8_t Rh_add_br = 32, int8_t rRh = 32,
        int8_t one_minus_update = 8,
        int8_t old_contrib = 32, int8_t new_contrib = 32
    ) {
        OperatorQuantConfig config;
        config.x_bitwidth = static_cast<QuantBitWidth>(x);
        config.h_bitwidth = static_cast<QuantBitWidth>(h);
        config.W_bitwidth = static_cast<QuantBitWidth>(W);
        config.R_bitwidth = static_cast<QuantBitWidth>(R);
        config.bx_bitwidth = static_cast<QuantBitWidth>(bx);
        config.br_bitwidth = static_cast<QuantBitWidth>(br);
        config.Wx_bitwidth = static_cast<QuantBitWidth>(Wx);
        config.Rh_bitwidth = static_cast<QuantBitWidth>(Rh);
        config.z_pre_bitwidth = static_cast<QuantBitWidth>(z_pre);
        config.z_out_bitwidth = static_cast<QuantBitWidth>(z_out);
        config.r_pre_bitwidth = static_cast<QuantBitWidth>(r_pre);
        config.r_out_bitwidth = static_cast<QuantBitWidth>(r_out);
        config.g_pre_bitwidth = static_cast<QuantBitWidth>(g_pre);
        config.g_out_bitwidth = static_cast<QuantBitWidth>(g_out);
        config.Rh_add_br_bitwidth = static_cast<QuantBitWidth>(Rh_add_br);
        config.rRh_bitwidth = static_cast<QuantBitWidth>(rRh);
        config.one_minus_update_bitwidth = static_cast<QuantBitWidth>(one_minus_update);
        config.old_contrib_bitwidth = static_cast<QuantBitWidth>(old_contrib);
        config.new_contrib_bitwidth = static_cast<QuantBitWidth>(new_contrib);
        return config;
    }

    // 预设配置：全 INT8
    static OperatorQuantConfig allInt8() {
        return OperatorQuantConfig();  // 默认就是全 INT8
    }

    // 预设配置：全 INT16
    static OperatorQuantConfig allInt16() {
        OperatorQuantConfig config;
        config.x_bitwidth = QuantBitWidth::INT16;
        config.h_bitwidth = QuantBitWidth::INT16;
        config.W_bitwidth = QuantBitWidth::INT16;
        config.R_bitwidth = QuantBitWidth::INT16;
        config.z_pre_bitwidth = QuantBitWidth::INT16;
        config.z_out_bitwidth = QuantBitWidth::INT16;
        config.r_pre_bitwidth = QuantBitWidth::INT16;
        config.r_out_bitwidth = QuantBitWidth::INT16;
        config.g_pre_bitwidth = QuantBitWidth::INT16;
        config.g_out_bitwidth = QuantBitWidth::INT16;
        config.one_minus_update_bitwidth = QuantBitWidth::INT16;
        return config;
    }

    // 预设配置：混合精度（门用 INT8，候选状态用 INT16）
    static OperatorQuantConfig mixedPrecision() {
        OperatorQuantConfig config;
        // 门控信号用 INT8（sigmoid 输出范围 [0,1]，精度需求较低）
        config.z_pre_bitwidth = QuantBitWidth::INT8;
        config.z_out_bitwidth = QuantBitWidth::INT8;
        config.r_pre_bitwidth = QuantBitWidth::INT8;
        config.r_out_bitwidth = QuantBitWidth::INT8;
        // 候选状态用 INT16（tanh 输出范围 [-1,1]，需要更高精度）
        config.g_pre_bitwidth = QuantBitWidth::INT16;
        config.g_out_bitwidth = QuantBitWidth::INT16;
        return config;
    }

    // 验证配置合理性
    bool validate() const {
        // 检查位宽是否为有效值
        auto isValidBitWidth = [](QuantBitWidth bw) {
            int8_t val = static_cast<int8_t>(bw);
            return val == 8 || val == 16 || val == 32 || val == -8 || val == -16;
        };

        return isValidBitWidth(x_bitwidth) &&
               isValidBitWidth(h_bitwidth) &&
               isValidBitWidth(W_bitwidth) &&
               isValidBitWidth(R_bitwidth) &&
               isValidBitWidth(bx_bitwidth) &&
               isValidBitWidth(br_bitwidth) &&
               isValidBitWidth(Wx_bitwidth) &&
               isValidBitWidth(Rh_bitwidth) &&
               isValidBitWidth(z_pre_bitwidth) &&
               isValidBitWidth(z_out_bitwidth) &&
               isValidBitWidth(r_pre_bitwidth) &&
               isValidBitWidth(r_out_bitwidth) &&
               isValidBitWidth(g_pre_bitwidth) &&
               isValidBitWidth(g_out_bitwidth) &&
               isValidBitWidth(Rh_add_br_bitwidth) &&
               isValidBitWidth(rRh_bitwidth) &&
               isValidBitWidth(one_minus_update_bitwidth) &&
               isValidBitWidth(old_contrib_bitwidth) &&
               isValidBitWidth(new_contrib_bitwidth);
    }

    // 获取各算子的运行时信息
    RuntimeBitWidthInfo getXInfo() const { return RuntimeBitWidthInfo::fromBitWidth(x_bitwidth); }
    RuntimeBitWidthInfo getHInfo() const { return RuntimeBitWidthInfo::fromBitWidth(h_bitwidth); }
    RuntimeBitWidthInfo getWInfo() const { return RuntimeBitWidthInfo::fromBitWidth(W_bitwidth); }
    RuntimeBitWidthInfo getRInfo() const { return RuntimeBitWidthInfo::fromBitWidth(R_bitwidth); }
};

// ==================== 量化操作辅助宏/工具 ====================

/**
 * @brief 根据位宽配置调用量化函数的宏
 *
 * 使用示例：
 * ```cpp
 * DISPATCH_QUANT_TYPE(config.x_bitwidth, QuantT, {
 *     quantize<QuantT>(data, quant_data, size, exp2_inv, zp);
 * });
 * ```
 */
#define DISPATCH_QUANT_TYPE(bitwidth, TypeName, ...)                           \
    do {                                                                        \
        dispatchByBitWidth(bitwidth, [&](auto type_tag) {                      \
            using TypeName = typename decltype(type_tag)::type;                \
            __VA_ARGS__                                                         \
            return 0;                                                           \
        });                                                                     \
    } while (0)

// ==================== 辅助函数 ====================

// 获取位宽的字符串表示
inline const char* bitWidthToString(QuantBitWidth bw) {
    switch (bw) {
        case QuantBitWidth::INT8:   return "INT8";
        case QuantBitWidth::INT16:  return "INT16";
        case QuantBitWidth::INT32:  return "INT32";
        case QuantBitWidth::UINT8:  return "UINT8";
        case QuantBitWidth::UINT16: return "UINT16";
        default:                    return "UNKNOWN";
    }
}

// 从字符串解析位宽
inline QuantBitWidth stringToBitWidth(const std::string& str) {
    if (str == "INT8" || str == "int8")   return QuantBitWidth::INT8;
    if (str == "INT16" || str == "int16") return QuantBitWidth::INT16;
    if (str == "INT32" || str == "int32") return QuantBitWidth::INT32;
    if (str == "UINT8" || str == "uint8") return QuantBitWidth::UINT8;
    if (str == "UINT16" || str == "uint16") return QuantBitWidth::UINT16;
    throw std::invalid_argument("Unknown bitwidth string: " + str);
}

// 获取位宽对应的字节大小
inline size_t bitWidthToByteSize(QuantBitWidth bw) {
    return RuntimeBitWidthInfo::fromBitWidth(bw).byte_size;
}
