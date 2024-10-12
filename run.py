from libs.pipeline import Pipeline

apps = Pipeline(["a"], {"pgie_config_path": "configs/pgies/yolov5.txt"})
apps.run()
