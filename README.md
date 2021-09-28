# ml_in_business
course project&amp;hw
Датасет с kaggle - https://www.kaggle.com/pinakimishrads/fake-news-classifier-data
Задача бинарной класификации является ли публикация фэйком или нет
Признаки:
1. title
2. text
Модель: GradientBoostingClassifier
Клонируем репозиторий и создаем образ
$ git clone https://github.com/pranalex666/ml_in_business.git
$ docker build . --tag <name>
Запускаем контейнер
$ docker run -d -p 8180:8180 -v <your_local_path_to_pretrained_models>:/app/app/models <name>
Запускаем скрипт 
Step_3.1.ipynb
