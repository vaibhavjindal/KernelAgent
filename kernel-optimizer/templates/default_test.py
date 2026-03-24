import inspect
import torch
import sys
sys.path.insert(0, ".")
from kernel import kernel_function
from problem import Model, get_inputs, get_init_inputs

def test_kernel():
    device = "cuda"
    dtype = torch.bfloat16

    # Build reference model
    init_inputs = get_init_inputs()
    if not isinstance(init_inputs, (tuple, list)):
        init_inputs = [init_inputs]
    model = Model(*init_inputs).to(device).to(dtype)

    # Prepare inputs
    raw_inputs = get_inputs()
    if not isinstance(raw_inputs, (tuple, list)):
        raw_inputs = [raw_inputs]
    inputs = [
        x.to(device).to(dtype) if isinstance(x, torch.Tensor) and x.is_floating_point()
        else x.to(device) if isinstance(x, torch.Tensor)
        else x
        for x in raw_inputs
    ]

    # Reference output
    with torch.no_grad():
        ref = model(*inputs)

    # Detect if kernel_function needs model parameters (weight, bias, etc.)
    _MODEL_PARAM_NAMES = {"weight", "w", "bias", "conv_bias", "eps",
                          "kernel_size", "stride", "padding", "dilation",
                          "groups", "num_groups", "normalized_shape"}
    needs_model = False
    kernel_params = []
    has_var_positional = False
    try:
        sig = inspect.signature(kernel_function)
        kernel_params = [
            name for name, p in sig.parameters.items()
            if p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD)
        ]
        has_var_positional = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL
            for p in sig.parameters.values()
        )
        if _MODEL_PARAM_NAMES.intersection(kernel_params):
            needs_model = True
        if not needs_model and has_var_positional:
            try:
                src = inspect.getsource(kernel_function)
                needs_model = any(kw in src for kw in ("weight", "w.shape", "kernel_size"))
            except (OSError, TypeError):
                pass
    except Exception:
        pass

    if needs_model:
        # Extract model parameters and build kernel args
        model_params = {}
        all_weights = []
        _CONV = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                 torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)
        _NORM = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                 torch.nn.LayerNorm, torch.nn.GroupNorm,
                 torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d)
        _POOL = (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d,
                 torch.nn.AvgPool1d, torch.nn.AvgPool2d, torch.nn.AvgPool3d,
                 torch.nn.AdaptiveAvgPool1d, torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveAvgPool3d,
                 torch.nn.AdaptiveMaxPool1d, torch.nn.AdaptiveMaxPool2d, torch.nn.AdaptiveMaxPool3d)
        for _, m in model.named_modules():
            if isinstance(m, (*_CONV, torch.nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None:
                    all_weights.append(m.weight)
                    model_params.setdefault("weight", m.weight)
                    model_params.setdefault("w", m.weight)
                    if getattr(m, "bias", None) is not None:
                        model_params.setdefault("conv_bias", m.bias)
                        model_params.setdefault("bias", m.bias)
                    for attr in ("stride", "padding", "dilation", "output_padding"):
                        val = getattr(m, attr, None)
                        if val is not None:
                            model_params.setdefault(attr, val)
                    if hasattr(m, "groups"):
                        model_params.setdefault("groups", m.groups)
            elif isinstance(m, _NORM):
                if getattr(m, "weight", None) is not None:
                    model_params.setdefault("weight", m.weight)
                    model_params.setdefault("w", m.weight)
                if getattr(m, "bias", None) is not None:
                    model_params.setdefault("bias", m.bias)
                if hasattr(m, "eps"):
                    model_params["eps"] = m.eps
                if hasattr(m, "num_groups"):
                    model_params["num_groups"] = m.num_groups
                if hasattr(m, "normalized_shape"):
                    model_params["normalized_shape"] = m.normalized_shape
            elif isinstance(m, _POOL):
                for attr in ("kernel_size", "stride", "padding", "dilation"):
                    val = getattr(m, attr, None)
                    if val is not None:
                        model_params.setdefault(attr, val)

        # Top-level bias on model itself (fusion kernels like Conv+ReLU+BiasAdd)
        if hasattr(model, "bias") and isinstance(model.bias, (torch.Tensor, torch.nn.Parameter)):
            model_params["add_bias"] = model.bias.to(device).to(dtype) if model.bias.is_floating_point() else model.bias.to(device)
            model_params.setdefault("bias", model_params["add_bias"])

        if has_var_positional and all_weights:
            # *args style: pass inputs + weights positionally, config as kwargs
            pos_args = list(inputs) + list(all_weights)
            config_kwargs = {}
            for k, v in model_params.items():
                if k not in ("weight", "w", "bias", "conv_bias", "add_bias"):
                    if isinstance(v, (tuple, list)) and len(v) >= 1 and all(e == v[0] for e in v):
                        v = v[0]
                    config_kwargs[k] = v
            out = kernel_function(*pos_args, **config_kwargs)
        else:
            # Named-parameter style
            bound = {}
            pos_idx = 0
            for pname in kernel_params:
                if pname in model_params:
                    v = model_params[pname]
                    if isinstance(v, (tuple, list)) and len(v) >= 1 and all(e == v[0] for e in v):
                        v = v[0]
                    bound[pname] = v
                elif pos_idx < len(inputs):
                    bound[pname] = inputs[pos_idx]
                    pos_idx += 1
            out = kernel_function(**bound)
    else:
        out = kernel_function(*inputs)

    if torch.allclose(ref, out, rtol=1e-2, atol=1e-2):
        print("PASS")
        return True
    else:
        diff = (ref - out).abs().max().item()
        print(f"FAIL: max diff = {diff}")
        return False

if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)
