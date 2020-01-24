# Algoritmo para detectar fraudes en tarjetas de crédito 

## Intro

Este modelo se basa en un modelo de aprendizaje supervisado.

Obtención de Data:
El Dataset es obtenido de [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) 

Los datos son dos días de un banco europeo donde hay mas de 280,000 transacciones, sin embargo, por la naturaleza de la información, los datos que están en el dataset son el resultado de PCA ya que de este modo la información que obtengamos será anónima, sin afectar la información en sí.

### Modelos Utilizados
Se utiliza las siguientes librería:
- **RandomForest**
- Pandas
- Numpy
- SKlearn

## Modelo

Lo primero que notamos al abrir el Dataset es el gran des balanceo de los datos pues solo 492 de 284,315 registros son fraudes esto es aprox .173% de los datos.

Podemos ver que se cuenta con 30 dimensiones o características, 28 las cuales fueron procesadas por PCA, el **monto de Transacción** y el **Tiempo** este último cuenta los segundos a partir de la primera transacción.
Todos los datos son del tipo float64 a excepción de Class(la etiqueta) que esta a INT64, por lo que podremos modificar el tipo de datos con el fin de hacer más optimo el procesamiento de datos.

Para trabajar con el des balanceo aplicaremos "Scale" de la librería SKlearn y aplicaremos PCA para reducirlo a dos dimensiones, posteriormente hacemos un split del 80 / 20 para datos de entrenamiento y testeo, esto en nuestro modelo de RandomForest y se realiza la predicción.

Para la evaluación del algoritmo aplicamos la métrica de Accuracy, la cual de un : **99.94382219725431%** de efectividad, sin embargo, esta métrica en este caso en especial es demás engañosa ya que nuestra muestra de Fraudes es de apenas .173% de los datos por lo que aplicaremos otra métrica, "roc_auc_score" ambas de Sklearn y ahora podemos ver que nuestro algoritmo tiene una efectividad de **50.98482298591267%** una métrica mucho más razonable viendo nuestra matriz de confusión

![Matriz](https://github.com/rogerzadi/Fraud_Detection_credit_card/blob/master/images/conf.png)

podemos ver que nuestro algoritmo a acertado en el 100% de las transacciones normales.

### Variables más importantes

Una ventaja al usar randomForest es que podemos ver las variables que tienen más peso 

![Tabla](https://github.com/rogerzadi/Fraud_Detection_credit_card/blob/master/images/impvar.png)

![Gra](https://github.com/rogerzadi/Fraud_Detection_credit_card/blob/master/images/impgra.png)

Al venir la info después de un proceso de PCA no podemos ver el nombre de las columnas pero con información interna aquí se podría ver las variables que tendrían más peso.

Cabe recalcar que este es un primer paso para nuestro algoritmo, donde posteriormente podremos utilizar más técnicas para mejorar el porcentaje, posteriormente subiré la actualización
