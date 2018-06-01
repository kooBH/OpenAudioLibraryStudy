#!/bin/sh
sudo echo 녹음 장치를 탐색합니다.
./audioprobe

echo "사용할 녹음 장치는 몇번 장치입니까? (위에서 0번부터 시작)"
read device
while true
do
	echo "$device 번 장치가 맞습니까?[Y/N](대문자만)"
	read ans
	if [ "$ans" = "Y" ];
	then
		break
	else
		echo "사용활 녹음 장치는 몇번 장치입니까? (위에서 0번부터시작)"
		read device
	fi
done
echo $device 번 장치를 사용합니다.

echo 몇 채널로 녹음합니까?
read channels

while true
do
	echo "$channels 가 맞습니까?[Y/N](대문자만)"
	read ans
	if [ "$ans" = "Y" ];
	then
		break
	else
		echo 몇 채널로 녹음합니까?
	 	read channels
	fi
done


echo 샘플 레이트는 몇 입니까?
read rates

while true
do
	echo "$rates 가 맞습니까?[Y/N](대문자만)"	
	read ans
	if [ "$ans" = "Y" ];
	then
		break
	else
		echo 샘플 레이트는 몇입니까?
		read rates
	fi
done


#sudo fdisk -l
df -h
echo "사용할 저장 장치의 경로(Mounted on)를 입력해 주십시오."
echo "예)/media/user/Device_Name (경로 상의 공백은 space를 누르면 됩니다.) "
read path
#df -h | grep $disk | cat 

while true
do
	if [ -d "$path" ]
		then
			echo "$path" 가 존재합니다.
			if [ ! -d "$path"/wavs ]
			then	
				sudo mkdir "$path"/wavs
				echo "$path"/wavs 폴더를 만들었습니다.
			fi	
			break;
		else
			echo "$path" 는 없습니다.
			echo "경로가 존재하지 않습니다. 다시 입력해주세요."
			read path
	fi
done
echo $path 를 사용합니다.

sudo chmod 777 "$path"/wavs

echo "startup application을 설정합니다.(부팅 시 자동실행되는 스크립트)"

echo "#!/bin/sh\npath=\"$path\"\nsleep 6\nif [ -d \"\$path\" ]\nthen\nif [ ! -d \"\$path\"/wavs ]\nthen\nmkdir \"\$path\"/wavs\nfi\ngnome-terminal -x 'bash' -c '$PWD/monitor \"$path\" $device $rates $channels 60 \"$PWD\";bash'\nelse\necho 경로가 존재하지 않습니다. setting을 다시 해주세요.\nfi ">rt_run.sh
sudo chmod +x rt_run.sh
if [ ! -d ~/.config/autostart ]
then
	mkdir ~/.config/autostart
fi

echo "[Desktop Entry]\nType=Application\nExec=$PWD/rt_run.sh\nHidden=false\nNoDisplay=false\nX-GNOME-Autostart-enabled=true\nName[en_US]=rt_run\nName=rt_run\nComment[en_US]=\nComment=" > ~/.config/autostart/rt_run.sh.desktop

echo "rt_run.sh 가 startup application으로 등록되었습니다.\n현재 환경과 동일한 환경으로 부팅시 실행됩니다.\n 변경된 사항이 있으면 제대로 동작하지 않을 수 있습니다.\n 환경이 바뀌었을 경우엔 setting.sh를 다시 실행해 주세요."

./monitor "$path" $device $rates $channels 60 "$PWD" 

