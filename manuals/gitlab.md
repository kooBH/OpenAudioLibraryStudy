Documnet : https://docs.gitlab.com/ce/README.html  

http://www.redmine.org/projects/redmine/wiki/HowTo_setup_automatic_refresh_of_repositories_in_Redmine_on_commit



## Environment

+ Ubuntu 16.04 위에 있는 Docker container

https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-gitlab-on-ubuntu-16-04

```bash

    sudo apt-get update
    sudo apt-get install ca-certificates curl openssh-server # postfix

    cd /tmp
    curl -LO https://packages.gitlab.com/install/repositories/gitlab/gitlab-ce/script.deb.sh

    bash /tmp/script.deb.sh
```
