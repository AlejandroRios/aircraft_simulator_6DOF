import re
import pandas as pd
import matplotlib.pyplot as plt


class DATCOMResultReader:
    def __init__(self, filepath):
        """
        Classe para ler os resultados do arquivo for006.dat do Missile DATCOM.
        """
        self.filepath = filepath
        self.dataframes = []  # Lista para armazenar os DataFrames

    def read_file(self):
        """
        Lê e processa o arquivo for006.dat.
        """
        with open(self.filepath, "r") as f:
            lines = f.readlines()

        # Dividir o arquivo em blocos separados por "CASE"
        blocks = self._split_cases(lines)
        for case_number, block in enumerate(blocks, 1):
            print(f"Processando CASE {case_number}...")
            self._process_case(block)

    def _split_cases(self, lines):
        """
        Divide o arquivo em blocos, cada um começando com "CASE".
        """
        cases = []
        current_case = []

        for line in lines:
            if "CASE" in line.upper():
                if current_case:
                    cases.append(current_case)
                current_case = [line]
            else:
                current_case.append(line)

        if current_case:
            cases.append(current_case)

        return cases

    def _process_case(self, block):
        """
        Processa um bloco correspondente a um caso específico.
        """
        conditions = self._extract_conditions(block)
        longitudinal_data, longitudinal_headers = self._extract_table(block, "LONGITUDINAL")
        derivatives_data, derivatives_headers = self._extract_table(block, "DERIVATIVES")

        if longitudinal_data:
            long_df = pd.DataFrame(longitudinal_data, columns=longitudinal_headers)
            for key, value in conditions.items():
                long_df[key] = value
            self.dataframes.append(("LONGITUDINAL", long_df))

        if derivatives_data:
            deriv_df = pd.DataFrame(derivatives_data, columns=derivatives_headers)
            for key, value in conditions.items():
                deriv_df[key] = value
            self.dataframes.append(("DERIVATIVES", deriv_df))

    def _extract_conditions(self, block):
        """
        Extrai as condições de voo de um bloco.
        """
        conditions = {}
        for line in block:
            if "MACH NO" in line.upper():
                match = re.search(r"MACH NO\s*=\s*([\d.]+)", line)
                if match:
                    conditions["MACH"] = float(match.group(1))
            elif "ALTITUDE" in line.upper():
                match = re.search(r"ALTITUDE\s*=\s*([\d.]+)\s*FT", line)
                if match:
                    conditions["ALTITUDE"] = float(match.group(1))
        print(f"Condições extraídas: {conditions}")
        return conditions

    def _extract_table(self, block, section_name):
        """
        Extrai uma tabela de dados de um bloco, baseado no nome da seção (ex: LONGITUDINAL).
        """
        data = []
        headers = []
        inside_table = False

        for line in block:
            if section_name in line.upper():
                inside_table = True
                continue

            if inside_table:
                if re.match(r"^\s*ALPHA", line):  # Cabeçalhos encontrados
                    headers = re.split(r"\s{2,}", line.strip())
                elif re.match(r"^\s*\d", line):  # Linhas de dados
                    values = re.split(r"\s{2,}", line.strip())
                    data.append(values)
                elif "CASE" in line.upper() or "-----" in line.upper():  # Fim da seção
                    break

        return data, headers

    def get_dataframes(self):
        """
        Retorna todos os DataFrames.
        """
        return self.dataframes


class DATCOMResultPlotter:
    def __init__(self, dataframes):
        """
        Classe para plotar os resultados processados.
        """
        self.dataframes = dataframes

    def plot(self, data_type, x_var, y_var):
        """
        Plota uma variável em relação a outra.

        Parâmetros:
        - data_type: Tipo de dados ("LONGITUDINAL" ou "DERIVATIVES").
        - x_var: Variável no eixo X (e.g., "ALPHA").
        - y_var: Variável no eixo Y (e.g., "CL").
        """
        for i, (dtype, df) in enumerate(self.dataframes):
            if dtype == data_type:
                plt.figure(figsize=(10, 6))
                plt.plot(pd.to_numeric(df[x_var]), pd.to_numeric(df[y_var]), marker="o", label=f"Case {i // 2 + 1}")
                plt.xlabel(x_var)
                plt.ylabel(y_var)
                plt.title(f"{y_var} vs {x_var} ({data_type})")
                plt.grid(True, linestyle="--", alpha=0.7)
                plt.legend()
        plt.show()


# Exemplo de uso
reader = DATCOMResultReader("for006_example.dat")
reader.read_file()

# Obter DataFrames
dataframes = reader.get_dataframes()

# Criar o plotter e plotar os resultados
plotter = DATCOMResultPlotter(dataframes)

# Plotar CL vs ALPHA
plotter.plot("LONGITUDINAL", x_var="ALPHA", y_var="CL")

# Plotar CD vs ALPHA
plotter.plot("LONGITUDINAL", x_var="ALPHA", y_var="CD")
