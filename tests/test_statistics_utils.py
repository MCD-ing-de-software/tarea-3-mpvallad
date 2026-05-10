import numpy as np
import numpy.testing as npt
import unittest

from src.statistics_utils import StatisticsUtils


class TestStatisticsUtils(unittest.TestCase):
    """Test suite for StatisticsUtils class."""

    def test_example_moving_average_with_numpy_testing(self):
        """Ejemplo de test usando numpy.testing para comparar arrays de NumPy.
        
        Este test demuestra cómo usar numpy.testing.assert_allclose() para comparar
        arrays de NumPy con tolerancia para errores de punto flotante, lo cual es
        esencial cuando trabajamos con operaciones numéricas.
        """
        utils = StatisticsUtils()
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = utils.moving_average(arr, window=3)
        
        # Valores esperados para media móvil con window=3
        expected = np.array([2.0, 3.0, 4.0])
        
        # Usar numpy.testing.assert_allclose() para comparar arrays de NumPy
        # Esto maneja correctamente errores de punto flotante con tolerancia
        npt.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

    def test_example_min_max_scale_with_numpy_testing(self):
        """Ejemplo de test usando numpy.testing para verificar transformaciones numéricas.
        
        Este test demuestra cómo usar numpy.testing.assert_allclose() para verificar
        que una transformación numérica produce los resultados correctos en todo el array,
        considerando errores de punto flotante en cálculos matemáticos.
        """
        utils = StatisticsUtils()
        arr = [10.0, 20.0, 30.0, 40.0]
        result = utils.min_max_scale(arr)
        
        # Valores esperados después de min-max scaling: (x - min) / (max - min)
        # min=10, max=40, range=30
        # [10->0.0, 20->0.333..., 30->0.666..., 40->1.0]
        expected = np.array([0.0, 1/3, 2/3, 1.0])
        
        # Usar numpy.testing.assert_allclose() para comparar arrays numéricos
        # La tolerancia relativa y absoluta permite errores pequeños de punto flotante
        npt.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_moving_average_basic_case(self):
        """Test que verifica que el método moving_average calcula correctamente la media móvil
        de una secuencia numérica para un caso básico.
        
        Escenario esperado:
        - Crear una lista de números (ej: [1, 2, 3, 4])
        - Llamar a moving_average con window=2
        - Verificar que el resultado es correcto (ej: [1.5, 2.5, 3.5] para el array dado) (usar numpy.testing.assert_allclose() para comparar arrays de NumPy - esto es mejor que unittest porque maneja la comparación de arrays numéricos con tolerancia para errores de punto flotante)
        - Verificar que el resultado tiene la forma (shape) esperada (usar self.assertEqual para comparar tuplas de .shape - comparación simple, unittest es suficiente)
        """

        # Creamos un objeto de la clase StatisticsUtils.
        # Este objeto nos permite usar el método moving_average.
        utils = StatisticsUtils()

        # Creamos una lista simple de números.
        # Esta será la entrada para calcular la media móvil.
        arr = [1, 2, 3, 4]

        # Llamamos al método moving_average.
        # window=2 significa que calculará promedios usando grupos de 2 números.
        result = utils.moving_average(arr, window=2)

        # Definimos el resultado que esperamos.
        # Promedio de [1, 2] = 1.5
        # Promedio de [2, 3] = 2.5
        # Promedio de [3, 4] = 3.5
        expected = np.array([1.5, 2.5, 3.5])

        # Comparamos el resultado real con el esperado.
        # Usamos assert_allclose porque son números decimales y puede haber pequeños errores de precisión.
        npt.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

        # Verificamos que la forma del resultado sea la misma que la del esperado.
        # En este caso ambos deberían tener 3 elementos.
        self.assertEqual(result.shape, expected.shape)

    def test_moving_average_raises_for_invalid_window(self):
        """Test que verifica que el método moving_average lanza un ValueError cuando
        se proporciona una ventana (window) inválida.
        
        Escenario esperado:
        - Crear una lista de números (ej: [1, 2, 3])
        - Llamar a moving_average con window=0 (valor no positivo) y verificar que se lanza un ValueError (usar self.assertRaises)
        - Llamar a moving_average con window mayor que la longitud del array y verificar que se lanza un ValueError (usar self.assertRaises)
        """

        # Creamos un objeto de StatisticsUtils para poder usar moving_average.
        utils = StatisticsUtils()

        # Creamos una lista de números de prueba.
        arr = [1, 2, 3]

        # Verificamos que se lance un ValueError.
        # window=0 es inválido porque la ventana debe ser mayor que cero.
        with self.assertRaises(ValueError):
            utils.moving_average(arr, window=0)

        # Verificamos otro caso inválido.
        # La lista tiene 3 elementos, entonces window=4 es demasiado grande.
        with self.assertRaises(ValueError):
            utils.moving_average(arr, window=4)

    def test_moving_average_only_accepts_1d_sequences(self):
        """Test que verifica que el método moving_average lanza un ValueError cuando
        se llama con una secuencia multidimensional.
        
        Escenario esperado:
        - Crear una secuencia bidimensional (ej: [[1, 2], [3, 4]])
        - Llamar a moving_average con esa secuencia y verificar que se lanza un ValueError indicando que solo se aceptan secuencias 1D (usar self.assertRaises)
        """

        # Creamos un objeto de StatisticsUtils.
        utils = StatisticsUtils()

        # Creamos una lista bidimensional.
        # No es una lista simple como [1, 2, 3].
        # Es una lista de listas: [[1, 2], [3, 4]].
        arr = [[1, 2], [3, 4]]

        # Verificamos que moving_average lance ValueError.
        # El método debería aceptar solo secuencias 1D, no matrices o listas 2D.
        with self.assertRaises(ValueError):
            utils.moving_average(arr, window=2)

    def test_zscore_has_mean_zero_and_unit_std(self):
        """Test que verifica que el método zscore calcula correctamente los z-scores
        de una secuencia numérica, comprobando que el resultado tiene media cero y
        desviación estándar unitaria.
        
        Escenario esperado:
        - Crear una lista de números (ej: [10, 20, 30, 40])
        - Llamar a zscore para obtener los z-scores (resultado es un array de NumPy)
        - Verificar que la media del resultado es aproximadamente 0 (usar self.assertAlmostEqual para un solo valor numérico - unittest es suficiente)
        - Verificar que la desviación estándar del resultado es aproximadamente 1 (usar self.assertAlmostEqual para un solo valor numérico - unittest es suficiente)
        """

        # Creamos un objeto de la clase StatisticsUtils.
        # Esto nos permite usar el método zscore.
        utils = StatisticsUtils()

        # Creamos una lista de números.
        # Esta será la entrada que vamos a transformar a z-scores.
        arr = [10, 20, 30, 40]

        # Llamamos al método zscore con la lista anterior.
        # El resultado debería ser un array de NumPy con los valores normalizados.
        result = utils.zscore(arr)

        # Verificamos que la media del resultado sea aproximadamente 0.
        # Usamos assertAlmostEqual porque puede haber pequeños errores decimales.
        # places=7 significa que comparamos hasta 7 decimales.
        self.assertAlmostEqual(result.mean(), 0.0, places=7)

        # Verificamos que la desviación estándar del resultado sea aproximadamente 1.
        # En un z-score correcto, la desviación estándar debe quedar igual a 1.
        self.assertAlmostEqual(result.std(), 1.0, places=7)

    def test_zscore_raises_for_zero_std(self):
        """Test que verifica que el método zscore lanza un ValueError cuando
        se llama con una secuencia que tiene desviación estándar cero
        (todos los valores son iguales).
        
        Escenario esperado:
        - Crear una lista con todos los valores iguales (ej: [5, 5, 5])
        - Llamar a zscore con esa secuencia y verificar que se lanza un ValueError indicando que la desviación estándar es cero (usar self.assertRaises)
        """

        # Creamos un objeto de la clase StatisticsUtils.
        # Esto nos permite usar el método zscore.
        utils = StatisticsUtils()

        # Creamos una lista donde todos los valores son iguales.
        # En este caso, la desviación estándar es 0.
        arr = [5, 5, 5]

        # Verificamos que zscore lance un ValueError.
        # No se puede calcular z-score si la desviación estándar es 0,
        # porque implicaría dividir por cero.
        with self.assertRaises(ValueError):
            utils.zscore(arr)

    def test_min_max_scale_maps_to_zero_one_range(self):
        """Test que verifica que el método min_max_scale escala correctamente una secuencia
        numérica al rango [0, 1], donde el valor mínimo se mapea a 0 y el máximo a 1.
        
        Escenario esperado:
        - Crear una lista de números (ej: [2, 4, 6])
        - Llamar a min_max_scale para obtener los valores escalados (resultado es un array de NumPy)
        - Verificar que el valor mínimo del resultado es 0.0 (usar self.assertAlmostEqual para un solo valor numérico - unittest es suficiente)
        - Verificar que el valor máximo del resultado es 1.0 (usar self.assertAlmostEqual para un solo valor numérico - unittest es suficiente)
        - Verificar que los valores transformados son correctos (ej: [0.0, 0.5, 1.0] para [2, 4, 6]) (usar numpy.testing.assert_allclose() para comparar el array completo - esto es necesario para comparar arrays de NumPy con tolerancia para errores de punto flotante)
        """

        # Creamos un objeto de la clase StatisticsUtils.
        # Esto nos permite usar el método min_max_scale.
        utils = StatisticsUtils()

        # Creamos una lista de números.
        # Queremos escalar estos valores al rango entre 0 y 1.
        arr = [2, 4, 6]

        # Llamamos al método min_max_scale.
        # Este método debería transformar el mínimo en 0.0,
        # el máximo en 1.0 y el valor intermedio en 0.5.
        result = utils.min_max_scale(arr)

        # Creamos el resultado esperado.
        # Para [2, 4, 6]:
        # 2 se convierte en 0.0
        # 4 se convierte en 0.5
        # 6 se convierte en 1.0
        expected = np.array([0.0, 0.5, 1.0])

        # Verificamos que el valor mínimo del resultado sea aproximadamente 0.0.
        # Usamos assertAlmostEqual porque trabajamos con números decimales.
        self.assertAlmostEqual(result.min(), 0.0, places=7)

        # Verificamos que el valor máximo del resultado sea aproximadamente 1.0.
        self.assertAlmostEqual(result.max(), 1.0, places=7)

        # Comparamos todo el array resultante con el array esperado.
        # Usamos numpy.testing.assert_allclose porque compara arrays numéricos
        # permitiendo pequeños errores de precisión decimal.
        npt.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

    def test_min_max_scale_raises_for_constant_values(self):
        """Test que verifica que el método min_max_scale lanza un ValueError cuando
        se llama con una secuencia donde todos los valores son iguales (no hay variación).
        
        Escenario esperado:
        - Crear una lista con todos los valores iguales (ej: [3, 3, 3])
        - Llamar a min_max_scale con esa secuencia y verificar que se lanza un ValueError indicando que todos los valores son iguales (usar self.assertRaises)
        """

        # Creamos un objeto de la clase StatisticsUtils.
        # Esto nos permite usar el método min_max_scale.
        utils = StatisticsUtils()

        # Creamos una lista donde todos los valores son iguales.
        # En este caso no hay mínimo y máximo distintos,
        # entonces no se puede escalar al rango [0, 1].
        arr = [3, 3, 3]

        # Verificamos que min_max_scale lance un ValueError.
        # Esto debe pasar porque si todos los valores son iguales,
        # habría una división por cero al hacer el escalamiento.
        with self.assertRaises(ValueError):
            utils.min_max_scale(arr)

if __name__ == "__main__":
    unittest.main()
