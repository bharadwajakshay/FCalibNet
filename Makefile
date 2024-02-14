clean:
	docker rm -f $$(docker ps -qa)
build:
	docker build --build-arg IMAGE_NAME=nvidia/cuda -t sampledocker .

run:
	docker run -it \
        --runtime=nvidia \
		-e NVIDIA_VISIBLE_DEVICES=0 \
		--ipc=host \
        --net=host \
        --privileged=true \
		--mount 'type=bind,source=${PWD}/checkpoints,target=/home/akshay/FCALIBNet/checkpoints' \
		--mount 'type=bind,source=${PWD}/logs,target=/home/akshay/FCALIBNet/logs'\
		--mount 'type=bind,source=/home/akshay/kitti_Dataset_40_1,target=/home/akshay/kitti_Dataset_40_1'\
		--mount 'type=bind,source=/mnt,target=/mnt' \
        --name="FcalibNetDocker" \
      	fcalibnet:unet3d_WEUC_NewData bash 