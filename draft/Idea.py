# nirdust

Contexto cientifico:

El continuo espectral de los AGN está compuesto en la banda K (centro 2.2 micrómetros) por 
la suma de el continuo estelar, el continuo producido por el disco de acreción (ley de potencias)
y una componente de tipo cuerpo negro o suma de cuerpos negros, la cual tiene su pico en 
3-5 micrómetros. Esta componente emerge en la emisión térmica de polvo calentado por el disco de 
acreción a temperaturas de 800-2000 K. (ver SED en el repo)

En particular, para los núcleos de tipo Seyfert 2, el continuo producido por el disco de acreción
no está presente. Por lo tanto, el continuo consiste solamente en la componente estelar mas la 
componente del polvo caliente.

Por lo tanto, en este tipo de nucleos es posible modelar mediante una funcion de cuerpo negro, o suma
de las mismas, la temperatura del polvo caliente. Tener en cuenta que antes de poder realizar el
ajuste debe ser sustraida la componente estelar.

-----------------------------

Estructura del código:

Este código toma como imput uno o mas continuos nucleares (es decir que previamente deben haber sido removidas 
las lineas espectrales). Por otro lado debe ser ingresado un espectro a un radio mayor donde el usuario
considera que el espectro es representativo de la población estelar de la región nuclear pero donde
la emisión del AGN ya no es significativa (entre 200 pc y 1000 pc anda bien para la mayoría de los casos
pero deberá ser evaluado por el usuario).

Pasa a frecuencias el eje espectral y normaliza el eje de flujo, tanto en los continuos como en las
funciones de cuerpo negro. Esto se hace para poder trabajar con espectros no calibrados en flujo,
es decir que sus unidades en el eje de intensidad son ADUs o unidades arbitrarias.

Luego se sustrae el continuo mas externo de los continuos nucleares para obtener el exceso 
producido por el polvo.

Finalmente se ajustan todos los continuos nucleares con funciones de cuerpo negro.

El output del código es en principio una distribución radial de temperaturas del polvo.
Pero, en caso de tratarse de un solo espectro nuclear el output puede ser simplemente el valor 
de la temperatura en ese punto radial.
Ademas, está la opción de obtener plots del exceso con el ajuste, como se ve en los plots que 
agrego en el repo.


Hay un código auxiliar, que se llama simulatorBB, que genera un dato sintético a partir de
una curva de cuerpo negro con ruido gaussiano y luego de aplicarle las mismas transformaciones que se les aplican
a los continuos en el código original, devuelve la misma temperatura ingresada dentro de un error 
aceptable. Ademas, este simulador sirvió para testear la estabilidad del ajuste: mediante la realización
de muchas iteraciones cambiando la temperatura inicial introducida se pudo ver que el ajuste no cambia,
es decir que es independiente de la condición inicial elegida. Respecto del error en la determinación
de la temperatura para el dato sintético, el simulador muestra que el error en el ajuste de
la temperatura es proporcional al ruido (gaussiano).

--------------------------------

Resultados obtenidos so far:

1) Ajuste de la temperatura del polvo para dos espectros nucleares de NGC 6300.
 https://ui.adsabs.harvard.edu/abs/2019AJ....157...44G/abstract 
 Pongo plot en el repo
 
 
2) Ajuste de la temperatura del polvo para 28 espectros nucleares de NGC 4945.
Ver plot de los ajustes en el repo y tambien plot de la distribucion radial de temperaturas.


--------------------------------

Problemas so far:

1) Por ahora solo lo he probado con dos set de espectros, ambos de Flamingos-2. 
   Y conseguir espectros de este tipo reducidos no es lo mas facil del mundo.
   
2) Por ahora los continuos libres de lineas espectrales los hago en IRAF. Sería un 
golazo que los haga el código así el usuario simplemente mete los .fits de sus espectros.

3) Hay que mejorar como se construye el vector 'radio', en caso de que el imput sea mas de un 
continuo.

4) Los errores por ahora son artesanales, los estimo viendo los limites de las temperaturas
que podrian ajustar cada espectro. Para automatizarlos hay que definir algún criterio
dependiente del ruido del espectro y de su "suavidad".
Tambien hay que establecer un criterio para el cual el ajuste falle, porque a veces 
los espectros estan muy defectuosos y el ajustador ajusta igual, dando una temperatura sin sentido.

5) Obviamente el estilo y la estructura en general de acuerdo a lo que fuimos aprendiendo en la materia.
