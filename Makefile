DOCKER_CMD := docker run -it --rm --gpus=all --privileged=true --ipc=host -v /media/hdd/models/docker:/opt/ml/models -v /media/ssd/workspace:/workspace -v /media/ssd/sspf-test-videos:/data
DOCKER_NSYS_CMD := ${DOCKER_CMD} --entrypoint=nsys
PROFILE_CMD := profile -t cuda,cublas,cudnn,nvtx,osrt --force-overwrite=true --delay=2 --duration=30

build-container: Dockerfile
	docker build -f $< -t deepstream-python:dev .


run-container: build-container
	${DOCKER_CMD} deepstream-python:dev

logs/%.qdrep: %.py
	${DOCKER_NSYS_CMD} deepstream-python:dev ${PROFILE_CMD} -o $@ python3 $<


