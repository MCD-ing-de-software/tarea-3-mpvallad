import pandas as pd
import pandas.testing as pdt
import unittest

from src.data_cleaner import DataCleaner


def make_sample_df() -> pd.DataFrame:
    """Create a small DataFrame for testing.

    The DataFrame intentionally contains missing values, extra whitespace
    in a text column, and an obvious numeric outlier.
    """
    return pd.DataFrame(
        {
            "name": [" Alice ", "Bob", None, " Carol  "],
            "age": [25, None, 35, 120],  # 120 is a likely outlier
            "city": ["SCL", "LPZ", "SCL", "LPZ"],
        }
    )


class TestDataCleaner(unittest.TestCase):
    """Test suite for DataCleaner class."""

    def test_example_trim_strings_with_pandas_testing(self):
        """Ejemplo de test usando pandas.testing para comparar DataFrames completos.
        
        Este test demuestra cómo usar pandas.testing.assert_frame_equal() para comparar
        DataFrames completos, lo cual es útil porque maneja correctamente los índices,
        tipos de datos y valores NaN de Pandas.
        """
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "Carol"],
            "age": [25, 30, 35]
        })
        cleaner = DataCleaner()
        
        result = cleaner.trim_strings(df, ["name"])
        
        # DataFrame esperado después de trim
        expected = pd.DataFrame({
            "name": ["Alice", "Bob", "Carol"],
            "age": [25, 30, 35]
        })
        
        # Usar pandas.testing.assert_frame_equal() para comparar DataFrames completos
        # Esto maneja correctamente índices, tipos y estructura de Pandas
        pdt.assert_frame_equal(result, expected)

    def test_example_drop_invalid_rows_with_pandas_testing(self):
        """Ejemplo de test usando pandas.testing para comparar Series.
        
        Este test demuestra cómo usar pandas.testing.assert_series_equal() para comparar
        Series completas, útil cuando queremos verificar que una columna completa tiene
        los valores esperados manteniendo los índices correctos.
        """
        df = pd.DataFrame({
            "name": ["Alice", None, "Bob"],
            "age": [25, 30, None],
            "city": ["SCL", "LPZ", "SCL"]
        })
        cleaner = DataCleaner()
        
        result = cleaner.drop_invalid_rows(df, ["name"])
        
        # Verificar que la columna 'name' ya no tiene valores faltantes
        # Los índices después de drop_invalid_rows son [0, 2] (se eliminó la fila 1)
        expected_name_series = pd.Series(["Alice", "Bob"], index=[0, 2], name="name")
        
        # Usar pandas.testing.assert_series_equal() para comparar Series completas
        # Esto verifica valores, índices y tipos correctamente
        pdt.assert_series_equal(result["name"], expected_name_series, check_names=True)

    def test_drop_invalid_rows_removes_rows_with_missing_values(self):
        """Test que verifica que el método drop_invalid_rows elimina correctamente las filas
        que contienen valores faltantes (NaN o None) en las columnas especificadas.
        
        Escenario esperado:
        - Crear un DataFrame con valores faltantes usando make_sample_df()
        - Llamar a drop_invalid_rows con las columnas "name" y "age"
        - Verificar que el DataFrame resultante no tiene valores faltantes en esas columnas (usar self.assertEqual para comparar .isna().sum() con 0 - comparación simple de enteros, unittest es suficiente)
        - Verificar que el DataFrame resultante tiene menos filas que el original (usar self.assertLess con len() - comparación simple de enteros, unittest es suficiente)
        """

        # Creamos un DataFrame de prueba usando la función auxiliar make_sample_df().
        # Este DataFrame tiene valores faltantes en las columnas "name" y "age".
        df = make_sample_df()

        # Creamos un objeto de la clase DataCleaner.
        # Esto nos permite usar el método drop_invalid_rows.
        cleaner = DataCleaner()

        # Llamamos al método drop_invalid_rows.
        # Le indicamos que elimine las filas que tengan valores faltantes
        # en las columnas "name" o "age".
        result = cleaner.drop_invalid_rows(df, ["name", "age"])

        # Verificamos que en la columna "name" ya no queden valores faltantes.
        # isna().sum() cuenta cuántos NaN o None hay.
        self.assertEqual(result["name"].isna().sum(), 0)

        # Verificamos que en la columna "age" ya no queden valores faltantes.
        self.assertEqual(result["age"].isna().sum(), 0)

        # Verificamos que el DataFrame resultante tenga menos filas que el original.
        # Esto confirma que efectivamente se eliminaron filas.
        self.assertLess(len(result), len(df))

    def test_drop_invalid_rows_raises_keyerror_for_unknown_column(self):
        """Test que verifica que el método drop_invalid_rows lanza un KeyError cuando
        se llama con una columna que no existe en el DataFrame.
        
        Escenario esperado:
        - Crear un DataFrame usando make_sample_df()
        - Llamar a drop_invalid_rows con una columna que no existe (ej: "does_not_exist")
        - Verificar que se lanza un KeyError (usar self.assertRaises)
        """

        # Creamos un DataFrame de prueba usando la función auxiliar make_sample_df().
        df = make_sample_df()

        # Creamos un objeto de la clase DataCleaner.
        # Esto nos permite usar el método drop_invalid_rows.
        cleaner = DataCleaner()

        # Verificamos que drop_invalid_rows lance un KeyError.
        # Esto debe ocurrir porque "does_not_exist" no existe como columna del DataFrame.
        with self.assertRaises(KeyError):
            cleaner.drop_invalid_rows(df, ["does_not_exist"])

    def test_trim_strings_strips_whitespace_without_changing_other_columns(self):
        """Test que verifica que el método trim_strings elimina correctamente los espacios
        en blanco al inicio y final de los valores en las columnas especificadas, sin modificar
        el DataFrame original ni las columnas no especificadas.
        
        Escenario esperado:
        - Crear un DataFrame con espacios en blanco usando make_sample_df()
        - Llamar a trim_strings con la columna "name"
        - Verificar que el DataFrame original no fue modificado (mantiene los espacios) (usar self.assertEqual para comparar valores específicos como strings individuales - unittest es suficiente para strings)
        - Verificar que en el DataFrame resultante los valores de "name" no tienen espacios al inicio/final (usar self.assertEqual para comparar valores específicos como strings individuales - unittest es suficiente)
        - Verificar que las columnas no especificadas (ej: "city") permanecen sin cambios (si comparas Series completas, usar pandas.testing.assert_series_equal() ya que maneja mejor los índices y tipos de Pandas; si comparas valores individuales, self.assertEqual es suficiente)
        """

        # Creamos un DataFrame de prueba usando la función auxiliar make_sample_df().
        # Este DataFrame tiene espacios en blanco en algunos valores de la columna "name".
        df = make_sample_df()

        # Creamos un objeto de la clase DataCleaner.
        # Esto nos permite usar el método trim_strings.
        cleaner = DataCleaner()

        # Llamamos al método trim_strings.
        # Le indicamos que limpie los espacios al inicio y al final
        # solamente en la columna "name".
        result = cleaner.trim_strings(df, ["name"])

        # Verificamos que el DataFrame original NO fue modificado.
        # En el DataFrame original, el primer nombre todavía debe tener espacios.
        self.assertEqual(df.loc[0, "name"], " Alice ")

        # También verificamos que el cuarto nombre del DataFrame original
        # conserve sus espacios originales.
        self.assertEqual(df.loc[3, "name"], " Carol  ")

        # Verificamos que en el DataFrame resultante se eliminaron los espacios
        # del primer nombre.
        self.assertEqual(result.loc[0, "name"], "Alice")

        # Verificamos que también se eliminaron los espacios del cuarto nombre.
        self.assertEqual(result.loc[3, "name"], "Carol")

        # Verificamos que una columna no especificada, como "city",
        # quedó igual en el resultado.
        # Usamos pandas.testing.assert_series_equal porque estamos comparando
        # una columna completa de Pandas.
        pdt.assert_series_equal(result["city"], df["city"], check_names=True)

    def test_trim_strings_raises_typeerror_for_non_string_column(self):
        """Test que verifica que el método trim_strings lanza un TypeError cuando
        se llama con una columna que no es de tipo string.
        
        Escenario esperado:
        - Crear un DataFrame usando make_sample_df()
        - Llamar a trim_strings con una columna numérica (ej: "age")
        - Verificar que se lanza un TypeError (usar self.assertRaises)
        """

        # Creamos un DataFrame de prueba usando la función auxiliar make_sample_df().
        df = make_sample_df()

        # Creamos un objeto de la clase DataCleaner.
        # Esto nos permite usar el método trim_strings.
        cleaner = DataCleaner()

        # Verificamos que trim_strings lance un TypeError.
        # Esto debe ocurrir porque "age" es una columna numérica,
        # y trim_strings debería usarse solo con columnas de texto.
        with self.assertRaises(TypeError):
            cleaner.trim_strings(df, ["age"])

    def test_remove_outliers_iqr_removes_extreme_values(self):
        """Test que verifica que el método remove_outliers_iqr elimina correctamente los
        valores extremos (outliers) de una columna numérica usando el método del rango
        intercuartílico (IQR).
        
        Escenario esperado:
        - Crear un DataFrame con valores extremos usando make_sample_df() (contiene edad=120)
        - Llamar a remove_outliers_iqr con la columna "age" y factor=1.5
        - Verificar que el valor extremo (120) fue eliminado del resultado (usar self.assertNotIn para verificar que 120 no está en los valores de la columna)
        - Verificar que al menos uno de los valores no extremos (25 o 35) permanece en el resultado (usar self.assertIn para verificar que está presente)
        """

        # Creamos un DataFrame de prueba con varios valores normales
        # y un valor extremo claro en la columna "age": 120.
        # Usamos más datos normales para que el método IQR pueda detectar mejor el outlier.
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Carol", "David", "Eva", "Frank"],
            "age": [25, 26, 27, 28, 29, 120],
            "city": ["SCL", "LPZ", "SCL", "LPZ", "SCL", "LPZ"],
        })

        # Creamos un objeto de la clase DataCleaner.
        # Esto nos permite usar el método remove_outliers_iqr.
        cleaner = DataCleaner()

        # Llamamos al método remove_outliers_iqr.
        # Le pedimos que elimine outliers en la columna "age"
        # usando factor=1.5, que es el valor típico del método IQR.
        result = cleaner.remove_outliers_iqr(df, "age", factor=1.5)

        # Verificamos que el valor extremo 120 ya no esté en la columna "age"
        # del DataFrame resultante.
        self.assertNotIn(120, result["age"].values)

        # Verificamos que al menos un valor no extremo permanezca.
        # En este caso, 25 debería seguir estando en la columna "age".
        self.assertIn(25, result["age"].values)

    def test_remove_outliers_iqr_raises_keyerror_for_missing_column(self):
        """Test que verifica que el método remove_outliers_iqr lanza un KeyError cuando
        se llama con una columna que no existe en el DataFrame.
        
        Escenario esperado:
        - Crear un DataFrame usando make_sample_df()
        - Llamar a remove_outliers_iqr con una columna que no existe (ej: "salary")
        - Verificar que se lanza un KeyError (usar self.assertRaises)
        """

        # Creamos un DataFrame de prueba usando la función auxiliar make_sample_df().
        df = make_sample_df()

        # Creamos un objeto de la clase DataCleaner.
        # Esto nos permite usar el método remove_outliers_iqr.
        cleaner = DataCleaner()

        # Verificamos que remove_outliers_iqr lance un KeyError.
        # Esto debe ocurrir porque "salary" no existe como columna del DataFrame.
        with self.assertRaises(KeyError):
            cleaner.remove_outliers_iqr(df, "salary")

    def test_remove_outliers_iqr_raises_typeerror_for_non_numeric_column(self):
        """Test que verifica que el método remove_outliers_iqr lanza un TypeError cuando
        se llama con una columna que no es de tipo numérico.
        
        Escenario esperado:
        - Crear un DataFrame usando make_sample_df()
        - Llamar a remove_outliers_iqr con una columna de texto (ej: "city")
        - Verificar que se lanza un TypeError (usar self.assertRaises)
        """

        # Creamos un DataFrame de prueba usando la función auxiliar make_sample_df().
        df = make_sample_df()

        # Creamos un objeto de la clase DataCleaner.
        # Esto nos permite usar el método remove_outliers_iqr.
        cleaner = DataCleaner()

        # Verificamos que remove_outliers_iqr lance un TypeError.
        # Esto debe ocurrir porque "city" es una columna de texto,
        # y el método remove_outliers_iqr solo debería trabajar con columnas numéricas.
        with self.assertRaises(TypeError):
            cleaner.remove_outliers_iqr(df, "city")

if __name__ == "__main__":
    unittest.main()
