# screenshot-vs-photo
Определение что на картинке: скриншот, фотография или картинка

### Требования
numpy
scipy
joblib
sklearn
keras
tensorflow
python3.6
### Обучение нейросети
```
python3 train.py
```
### Пример использования
```
python3 classifier.py
Расположение изображения:

image_examples/picture1.jpg
photo 	 0.0477564
picture 	 0.9501758
screenshot 	 0.0020678386
Результат: picture
```
```
Расположение изображения:
<путь к изображению>
```
 
 ### Файлы
 * image_examples   - примеры тестовых изображений
 * ./model.h5   - модель 
 * ./lb.sav   - кодировщик 
 * ./train.py    - скрипт обучения НС
 * ./classifier.py     - скрипт распознавания изображения
 * ./models.py - структура НС
 
 