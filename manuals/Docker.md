[초보를 위한 도커 안내서](https://subicura.com/2017/01/19/docker-guide-for-beginners-1.html)

sudo usermod -aG docker $USER # 현재 접속중인 사용자에게 권한주기
sudo usermod -aG docker your-user # your-user 사용자에게 권한주기



docker run [OPTIONS] IMAGE[:TAG|@DIGEST] [COMMAND] [ARG...]

다음은 자주 사용하는 옵션들입니다.
옵션 	설명
-d 	detached mode 흔히 말하는 백그라운드 모드
-p 	호스트와 컨테이너의 포트를 연결 (포워딩)
-v 	호스트와 컨테이너의 디렉토리를 연결 (마운트)
-e 	컨테이너 내에서 사용할 환경변수 설정
–name 	컨테이너 이름 설정
–rm 	프로세스 종료시 컨테이너 자동 제거
-it 	-i와 -t를 동시에 사용한 것으로 터미널 입력을 위한 옵션
–link 	컨테이너 연결 [컨테이너명:별칭]


https://hub.docker.com/r/sameersbn/redmine/

docker pull sameersbn/redmine

wget https://raw.githubusercontent.com/sameersbn/docker-redmine/master/docker-compose.yml
docker-compose up

