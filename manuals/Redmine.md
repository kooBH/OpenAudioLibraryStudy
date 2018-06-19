# [Redmine](../README.md)<a name = "TOP"></a>
1. [설치](#install)
2. [접속](#enter)
3. [git 연동](#git)

[Redmine](https://www.redmine.org/) is a flexible project management web application. Written using the Ruby on Rails framework, it is cross-platform and cross-database.    

[Redmine Guidle](https://www.redmine.org/projects/redmine/wiki/Guide)  
    
bitnami 를 이용한 편한 설치  
http://www.redmine.or.kr/projects/community/wiki/Linux

[bitnami](https://bitnami.com/)     
Bitnami has automated the ability to package, deploy and maintain applications, lowering the barrier to adoption for anyone to deploy and maintain a full spectrum of server applications, development stacks and infrastructure applications in virtually any format. 

## [설치](#TOP)<a name = "install"></a>

[The Bitnami Redmine Stack](https://bitnami.com/stack/redmine)   
Bitnami 는 다른 패키지를 많이 사용하는 어플리케이션을 이용할때, 한번에 패키지를 설치하거나 패키지가 설치된 가상머신을 사용할 수 있게한다. 어플리케이션 마다 다른 구성의 Bitnami를 지원한다   

1. 자신에 OS에 맞는 인스톨러를 받는다. 자신의 OS에 직접 설치할 수도 있고, 가상 머신을 사용할 수도 있다.
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
  

  
