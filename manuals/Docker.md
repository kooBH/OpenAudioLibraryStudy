# Docker 
도커는 VM 처럼 게스트 OS를 띄우지 않고  
Dcoker 위에 이미지들을 올리고 그 이미지 들을 엮어서 컨테이너를 만들 수 있게한다  
패키지 A B C D 가 있을 때, 이 들을 이미지로 만들고

A + B  
B + C  
A + C + D  

이렇게 사용하는 독립된 컨테이너를 만들어서 사용할 수 있다는 것  




## 쉬운 설치

https://hub.docker.com/r/sameersbn/redmine/

docker pull sameersbn/redmine

wget https://raw.githubusercontent.com/sameersbn/docker-redmine/master/docker-compose.yml
docker-compose up

+ 서버컴퓨터의 IP:10083 으로 접속가능
+ 호스트 OS의 /srv/docker/폴더는 docker 이미지에 마운트되어있다  
