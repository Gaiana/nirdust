NIRDUST

todas las instancias son de la clase NirdustSpectrum

1) Leer los espectros

nuclear_spectrum = nirdust.read_spectrum(path,extension)

external_spectrum = nirdust.read_spectrum(path, extension)



2) Corregir por z (INTERNO):
    
Esto debe ser un metodo interno de la clase. El z debe ser ingresado a mano
por el usuario.




3) cortar los bordes de los espectros: (es la funcion split que charlamos ayer)
    
sp_nuclear_spectrum = nuclear_spectrum.cut_edges(lmin,lmax) 
sp_external_spectrum = external_spectrum.cut_edges (lmin,lmax) 



4) remover lineas de espectros (INTERNO):
    
Usar specutils para que nirdust automaticamente  identifique y remueva las lineas
espectrales de ambos espectros, dejando solo el continuo.    
    
nuclear_spectrum = nirdust.remove_lines(espectro_nuclear)

external_spectrum = nirdust.remove_lines(espectro_externo)




5) Nirdustprepare (INTERNO):
    
Esta funcion recibe como parametros solo los dos objetos espectros.
Internamente pasa a frecuencia, normaliza ambos espectros y los resta.
    
def nirdustprepare(sp_nuclear_spectrum, sp_external_spectrum):
    ...
    
    return red_excess


6) Fittear el blackbody:

Se usa la funcion de Black Body de Astropy y algun metodo de ajuste de Astropy 
para ajustar el exceso resultante de Nirdustprepare.
    
dust_temperature, t_error = nirdust.dust_fitting(red_excess, initial_guess)


Nota: red_excess es introducido por nirdust, no por el usuario ya que sale de 
una funcion interna. Osea qeu dust_fitting tiene como unico parametro de entrada
la initial_guess.    

 
