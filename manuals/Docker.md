# [Docker](../README.md)<a name="TOP"></a> 
1. [컨테이너 백업](#backup)
2. [docker-compose](#compose)

도커는 VM 처럼 게스트 OS를 띄우지 않고  
Dcoker 위에 이미지들을 올리고 그 이미지 들을 엮어서 컨테이너를 만들 수 있게한다  
패키지 A B C D 가 있을 때, 이 들을 이미지로 만들고

A + B  
B + C  
A + C + D  

이렇게 사용하는 독립된 컨테이너를 만들어서 사용할 수 있다는 것  

또한 대부분의 패키지들이 이미지로 구현되어 있다 
이미지들을 받는 도커허브 https://hub.docker.com/explore/  

https://docs.docker.com/ 

## [컨테이너 백업](#TOP)<a name = "backup"></a>

```bash
$ docker container ls

CONTAINER ID        IMAGE                        COMMAND                  CREATED             STATUS              PORTS                            NAMES
e74c271093e8        sameersbn/redmine:3.4.6      "/sbin/entrypoint...."   About an hour ago   Up About an hour    443/tcp, 0.0.0.0:10083->80/tcp   ffe_redmine_1

$ docker commit <백업하려는 컨테이너의 ID> <백업이름>
ex)sudo docker commit e74c271093e8 redmine_bk_d21_1

$ docker images 

REPOSITORY             TAG                 IMAGE ID            CREATED             SIZE
redmine_bk_d21_1       latest              0fb473046406        10 seconds ago      961MB

```

## [docker-compose](#TOP)<a name="compose"></a>

docker 설정들을 저장해서 사용을 도와준다

+ 설치
```bash
$ sudo curl -L https://github.com/docker/compose/releases/download/1.21.2/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose
$ sudo chmod +x /usr/local/bin/docker-compose
```


https://docs.docker.com/compose/  




