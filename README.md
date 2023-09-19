# bsbz-ai-server

## 0. 시작하기에 앞서
pull 받으면 여기 있는 명령어 순서대로 한번씩 해주세요! 
이걸로 패키지랑 DB 세팅이랑 DB 입력까지 다하는 거니까 귀찮아하지 말아주세요..

## 1. Python & Package install
1. python 3.9.9 설치
2. 해당 폴더 위치에서 cmd 혹은 git bash 열기
3. `pip install -r requirements.txt`

## 2. Data migration
`python manage.py migrate`
`python process.py`

## 3. Server 가동 방법
`python manage.py runserver`