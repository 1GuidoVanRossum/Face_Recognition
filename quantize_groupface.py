import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from measure_weight_groupface import load_groupface_model, count_parameters, estimate_model_size, save_model_size


def quantize_model(model):
    # Применяем динамическое квантование для слоев Linear
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return quantized_model


def measure_quantized_model_weight():
    model = load_groupface_model()
    quant_model = quantize_model(model)
    num_params = count_parameters(quant_model)
    # Для квантованных параметров (qint8) считаем 1 байт на параметр
    estimated_mem = estimate_model_size(quant_model, dtype_size=1)
    disk_size = save_model_size(quant_model, "groupface_quantized.pth")

    print("=== Квантованная модель GroupFace ===")
    print(f"Параметров: {num_params:,}")
    print(f"Оценочный объем памяти (qint8): {estimated_mem:.2f} МБ")
    print(f"Размер на диске: {disk_size:.2f} МБ")

    return quant_model, num_params, estimated_mem, disk_size


if __name__ == "__main__":
    measure_quantized_model_weight()
