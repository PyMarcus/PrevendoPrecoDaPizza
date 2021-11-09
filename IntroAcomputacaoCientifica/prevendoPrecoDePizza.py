# modelo envolvendo machine learning 01
"""
 Prevendo o Preço da Pizza

Suponha que você queira prever o preço da pizza.
Para isso, vamos criar um modelo de regressão linear(pois, se quer prever um valor numérico)
para prever o preço da pizza, baseado em um atributo da pizza
que podemos observar.
Vamos modelar a relação entre o tamanho (diâmetro) de uma pizza
e seu preço. Escreveremos então um programa com sckit-learn,
que prevê o preço da pizza dado seu tamanho.

 """
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# tamanhos e precos de pizzas comidas, por exemplo
diametros = [[7], [10], [15], [30], [45]]
# Preços (R$)
precos = [[8], [11], [16], [38.5], [52]]

# análise exploaratória:
plt.figure()
plt.xlabel('Diâmetro(cm)')
plt.ylabel('Preço(R$)')
plt.title('Diâmetro x Preço')
plt.plot(diametros, precos, 'k.')
plt.axis([0, 60, 0, 60])
plt.grid(True)
plt.show()

# com base no gráfico, pode-se perceber que o diametro é diretamente proporcional ao preço
modelo = LinearRegression() # instanciação
print(type(modelo))

X = diametros
Y = precos
# treinar modelo
treinado = modelo.fit(X, Y)
print(treinado)

#previsao
prev = modelo.predict([[20], [0]])
preco = [preco[0] for preco in prev]

# prevendo preço de uma pizza de 20cm de diâmetro:
print(f"Uma pizza de 20cm de diâmetro deve custar cerca de {round(preco[0], 2)} reais")
