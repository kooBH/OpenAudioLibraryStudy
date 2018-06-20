# [Redmine](../README.md)<a name = "TOP"></a>
1. [설치](#install)
2. [접속](#enter)
3. [git 연동](#git)
4. [웹](#web)

[Redmine](https://www.redmine.org/) is a flexible project management web application. Written using the Ruby on Rails framework, it is cross-platform and cross-database.    

[Redmine Guidle](https://www.redmine.org/projects/redmine/wiki/Guide)  
    
# TODO
dcoker 로 redmine  
docker 설치는 함


## [설치](#TOP)<a name = "install"></a>

+ [Docker](https://www.docker.com/)  
설치  : http://pseg.or.kr/pseg/infoinstall/6067  

레드마인 : https://hub.docker.com/_/redmine/

도커 활용 : http://raccoonyy.github.io/docker-usages-for-dev-environment-setup/  

도커 버전 문서 :https://docs.docker.com/compose/compose-file/compose-versioning/#version-2


``` bash

$ sudo docker run -d --name redmine_mysql -e MYSQL_ROOT_PASSWORD=qwer1234 -e MYSQL_DATABASE=redmine_db mysql

$ sudo docker run -d --name some-redmine --link redmine_mysql:mysql redmine

$ curl -L https://github.com/docker/compose/releases/download/1.21.1/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose

$  sudo chmod +x /usr/local/bin/docker-compose 

$ sudo apt-get install docker-compose -- 이전 버전 설치함



$ vim docker-compose.yml
```
<details><summary>docker-compose.yml</summary>
    
```
version: '3.1'

services:

    redmine:
        image: redmine
        restart : always
        container_name: redmine
        ports:
            - 8080:3000
        environment:
            REDMINE_DB_MYSQL: db
            REDMINE_DB_PASSWORD: qwer1234

    db:
            #image: mysql # error | mbind : operation not permmited 
            image: mysql:5.7
            restart: always
            ports:
                - 3306:3306
            environment:
                MYSQL_ROOT_PASSWORD: qwer1234
                MYSQL_DATABASE: redmine

```    
</details>



``` bash
$ sudo docker-compose -f docer-compose.yml up

```
초기 주소 : http://localhost:8080
초기 관리자 계정 : Login : admin  |  Password : admin  


sudo docker run -it redmine bash

https://www.redmineup.com/pages/help/installation/installing-redmine-on-ubuntu-16-04

https://www.redmineup.com/pages/help/installation/installing-redmine-on-ubuntu-16-04#Setting-a-secure-connection-HTTPS-in-nginx

```bash
subversion imagemagick libmagickwand-dev libcurl4-openssl-dev curl

# gpg --keyserver hkp://keys.gnupg.net --recv-keys D39DC0E3
# curl -L https://get.rvm.io | bash -s stable --ruby=2.2.6

```

[The Bitnami Redmine Stack](https://bitnami.com/stack/redmine)   
Bitnami 는 다른 패키지를 많이 사용하는 어플리케이션을 이용할때, 한번에 패키지를 설치하거나 패키지가 설치된 가상머신을 사용할 수 있게한다. 어플리케이션 마다 다른 구성의 Bitnami를 지원한다   

자신에 OS에 맞는 인스톨러를 받는다. 자신의 OS에 직접 설치할 수도 있고, 가상 머신을 사용할 수도 있다.  
+ Ubuntu 인스톨러   
  1. 추가적으로 설치할 구성요소  
  2. 설치 경로
  3. 레드마인 계정
      * You real name
      * Email Address
      * Login
      * Password    
  4. 언어  
  5. 이메일로 Notification 가능? 
  6. Redmine을 클라우드에서 구동(유료)
  7. 완료  
 
  
bitnami 를 이용한 편한 설치  
http://www.redmine.or.kr/projects/community/wiki/Linux

[bitnami](https://bitnami.com/)     
Bitnami has automated the ability to package, deploy and maintain applications, lowering the barrier to adoption for anyone to deploy and maintain a full spectrum of server applications, development stacks and infrastructure applications in virtually any format. 

Bitnami 로 하면 원래의 redmine 구조가 아닌 bitnami의 구조로 설치가 되며, 설정또한 원래의 설정퍼일과 bitnami에서 관리하는 설정파일이 따로 있어서 일이 많다  


  ## [접속](#TOP)<a name = "enter"></a>
  
  설치 폴더에서 manager-linux-x64.run을 실행  
  Go to Application 으로 사이트에 접속 할 수 있다 
  거기서 Admin 계정으로 로그인하면 환경설정 가능  
  
  ## [git](#TOP)<a name = "git"></a>
  
  Redmine의 프로젝트를 나의 git 과 연결히려면  
  project -> Settings -> Reoisutirues -> New Repository  
  + SCM : git
  + Main repository : option (default yes)
  + Identifier
  + Path to repository : e.g /home/ffe/git/OpenAudioLibraryStudy/.git
  
  하면은 project에 Repository 항목이 생긴다  
  처음 열때는 이전 commit기록을 다 불러오기 때문에 시간이 좀 걸린다  
  
  ### Issue : Redmine 의 repo는 local git과 연동되어있다. github와의 sync를 위해 주기적으로 update를 하거나 pull하거나 우선순위를 정해서 처리하는걸 생각해 봐야함
  
  ## [WEB](#TOP)<a name = "web"></a> 
  
  http://matoker.com/30183738344
  
  libapache2-mod-passenger
 ruby-dev
 
 https://community.bitnami.com/t/redmine-how-could-i-change-the-localhost-8080-to-public-url-example-com/23362/18
  
