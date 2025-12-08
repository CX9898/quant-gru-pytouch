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
    INT32 = 32, // 用于中间累加
    UINT8 = -8, // 无符号 8 位
    UINT16 = -16// 无符号 16 位
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
inline auto dispatchByBitWidth(QuantBitWidth bw, Func &&func)
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

// ==================== 算子量化位宽配置结构 ====================
struct OperatorQuantConfig {
    QuantBitWidth x_bitwidth = QuantBitWidth::INT8;               // 输入 x
    QuantBitWidth h_bitwidth = QuantBitWidth::INT8;               // 隐藏状态 h
    QuantBitWidth W_bitwidth = QuantBitWidth::INT8;               // 权重 W
    QuantBitWidth R_bitwidth = QuantBitWidth::INT8;               // 权重 R
    QuantBitWidth bx_bitwidth = QuantBitWidth::INT8;             // 偏置 bx
    QuantBitWidth br_bitwidth = QuantBitWidth::INT8;             // 偏置 br
    QuantBitWidth Wx_bitwidth = QuantBitWidth::INT8;             // Wx 结果
    QuantBitWidth Rh_bitwidth = QuantBitWidth::INT8;             // Rh 结果
    QuantBitWidth z_pre_bitwidth = QuantBitWidth::INT8;           // z 门输入
    QuantBitWidth z_out_bitwidth = QuantBitWidth::INT8;           // z 门输出
    QuantBitWidth r_pre_bitwidth = QuantBitWidth::INT8;           // r 门输入
    QuantBitWidth r_out_bitwidth = QuantBitWidth::INT8;           // r 门输出
    QuantBitWidth g_pre_bitwidth = QuantBitWidth::INT8;           // g 门输入
    QuantBitWidth g_out_bitwidth = QuantBitWidth::INT8;           // g 门输出
    QuantBitWidth Rh_add_br_bitwidth = QuantBitWidth::INT8;      // Rh + br
    QuantBitWidth rRh_bitwidth = QuantBitWidth::INT8;            // r × Rh
    QuantBitWidth one_minus_update_bitwidth = QuantBitWidth::INT8;// 1 - z
    QuantBitWidth old_contrib_bitwidth = QuantBitWidth::INT8;    // z * h[output_idx]
    QuantBitWidth new_contrib_bitwidth = QuantBitWidth::INT8;    // (1.0 - z) * g
};
