PY ?= python

.PHONY: setup assets quantize test demo bench plot all

setup:            ## install dependencies
	$(PY) -m pip install -r requirements-dev.txt

assets:           ## download models + COCO128 sample data (~70 MB)
	$(PY) scripts/download_assets.py

quantize: assets  ## build INT8 models
	for m in yolox_nano yolox_tiny yolox_s; do $(PY) scripts/quantize.py models/$$m.onnx; done

test:             ## unit + integration tests
	$(PY) -m pytest -q

demo:             ## live webcam demo with the fastest accurate model
	$(PY) run_realtime.py --source 0 --model models/yolox_tiny.int8-static.onnx

bench: quantize   ## full benchmark on this machine (~10-20 min)
	$(PY) benchmark.py
	$(PY) scripts/plot_results.py results/$$(hostname)/benchmark.json

all: setup quantize test bench
