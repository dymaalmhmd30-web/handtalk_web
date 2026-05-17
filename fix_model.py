import h5py
import json
import shutil

def fix_model(input_model, output_model):
    shutil.copy(input_model, output_model)

    with h5py.File(output_model, "r+") as f:
        model_config = f.attrs.get("model_config")

        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")

        config = json.loads(model_config)

        def fix_item(item):
            if isinstance(item, dict):

                # إصلاح InputLayer
                if item.get("class_name") == "InputLayer":
                    cfg = item.get("config", {})

                    if "batch_shape" in cfg:
                        cfg["batch_input_shape"] = cfg.pop("batch_shape")

                    if "optional" in cfg:
                        cfg.pop("optional")

                # إصلاح DTypePolicy
                if "dtype" in item:
                    dtype_value = item["dtype"]

                    if isinstance(dtype_value, dict):
                        if dtype_value.get("class_name") == "DTypePolicy":
                            item["dtype"] = dtype_value.get("config", {}).get("name", "float32")

                for value in item.values():
                    fix_item(value)

            elif isinstance(item, list):
                for value in item:
                    fix_item(value)

        fix_item(config)

        f.attrs.modify("model_config", json.dumps(config).encode("utf-8"))

    print("Fixed:", output_model)


fix_model("arabic_model.h5", "arabic_model_fixed.h5")