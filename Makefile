pre:
	wget https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/mmcv_full-1.4.0-cp38-cp38-manylinux1_x86_64.whl
	python -m pip install mmcv_full-1.4.0-cp38-cp38-manylinux1_x86_64.whl
	python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
	mkdir -p thirdparty
	git clone https://github.com/open-mmlab/mmdetection.git thirdparty/mmdetection
	cd thirdparty/mmdetection && git checkout v2.16.0 && python -m pip install -e .
install:
	make pre
	python -m pip install -e .
clean:
	rm -rf thirdparty
	rm -r ssod.egg-info
