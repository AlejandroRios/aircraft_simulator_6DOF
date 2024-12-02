import matplotlib.pyplot as plt
import re


class MissileGeometryPlotterFromFile:
    def __init__(self, input_file):
        """
        Classe para plotar a geometria 2D do míssil a partir do arquivo de entrada gerado pelo MissileDATCOMInput.
        :param input_file: Nome do arquivo de entrada (string).
        """
        self.input_file = input_file
        self.geometry = {}

    def parse_input_file(self):
        """
        Lê e interpreta o arquivo de entrada para extrair informações geométricas.
        """
        with open(self.input_file, "r") as f:
            lines = f.readlines()

        # Variáveis para armazenar informações
        axibod_data = {}
        finset_data = []

        current_section = None

        for line in lines:
            line = line.strip()

            # Identificar início de uma seção
            if line.startswith("$AXIBOD"):
                current_section = "AXIBOD"
                continue
            elif line.startswith("$FINSET"):
                current_section = "FINSET"
                finset_data.append({})
                continue
            elif line.startswith("$"):
                current_section = None  # Fim de qualquer seção atual
                continue

            # Processar dados da seção atual
            if current_section == "AXIBOD":
                match = re.match(r"(\w+)=([\d.]+)", line)
                if match:
                    key, value = match.groups()
                    axibod_data[key] = float(value)
            elif current_section == "FINSET" and finset_data:
                match = re.match(r"(\w+)=([\d.,]+)", line)
                if match:
                    key, value = match.groups()
                    values = list(map(float, value.split(',')))
                    finset_data[-1][key] = values

        # Armazenar as informações extraídas
        self.geometry["AXIBOD"] = axibod_data
        self.geometry["FINSET"] = finset_data

    def plot_geometry(self):
        """
        Gera o plot 2D da geometria do míssil.
        """
        if not self.geometry:
            raise ValueError("A geometria não foi carregada. Execute 'parse_input_file()' primeiro.")

        # Dados do AXIBOD
        axibod = self.geometry["AXIBOD"]
        nose_length = axibod.get("LNOSE", 0)
        body_length = axibod.get("LCENTR", 0)
        nose_diameter = axibod.get("DNOSE", 0)
        body_diameter = nose_diameter  # Diâmetro constante do corpo principal

        total_length = nose_length + body_length

        # Configurações para o plot
        fig, ax = plt.subplots(figsize=(12, 4))

        # Plot do nariz (cone)
        cone_x = [0, nose_length]
        cone_y_upper = [0, body_diameter / 2]
        cone_y_lower = [0, -body_diameter / 2]
        ax.plot(cone_x, cone_y_upper, color="black")
        ax.plot(cone_x, cone_y_lower, color="black")

        # Plot do corpo (cilindro)
        x_body = [nose_length, total_length]
        y_body_upper = [body_diameter / 2, body_diameter / 2]
        y_body_lower = [-body_diameter / 2, -body_diameter / 2]
        ax.plot(x_body, y_body_upper, color="black")
        ax.plot(x_body, y_body_lower, color="black")

        # Fechar a traseira do cilindro
        ax.plot([total_length, total_length], [-body_diameter / 2, body_diameter / 2], color="black")

        # Dados das aletas (FINSET)
        for i, finset in enumerate(self.geometry["FINSET"], 1):
            fin_position = finset.get("XLE", [0])[0]  # Posição inicial da aleta ao longo do eixo X
            root_chord = finset.get("CHORD", [0, 0])[0]  # Comprimento da raiz da aleta
            tip_chord = finset.get("CHORD", [0, 0])[1] if len(finset.get("CHORD", [])) > 1 else 0  # Comprimento da ponta da aleta
            span = finset.get("SSPAN", [0])[-1]  # Comprimento máximo da envergadura

            # Coordenadas do trapézio ou triângulo
            fin_x = [
                fin_position,                     # Borda de ataque (raiz)
                fin_position + root_chord,        # Borda de fuga (raiz)
                fin_position + tip_chord,         # Borda de ataque (ponta)
            ]
            fin_y_upper = [body_diameter / 2, body_diameter / 2, body_diameter / 2 + span]
            fin_y_lower = [-body_diameter / 2, -body_diameter / 2, -body_diameter / 2 - span]

            # Fechar a forma corretamente
            fin_x_upper = [fin_x[0], fin_x[1], fin_x[2], fin_x[0]]
            fin_x_lower = [fin_x[0], fin_x[1], fin_x[2], fin_x[0]]
            fin_y_upper_closed = [fin_y_upper[0], fin_y_upper[1], fin_y_upper[2], fin_y_upper[0]]
            fin_y_lower_closed = [fin_y_lower[0], fin_y_lower[1], fin_y_lower[2], fin_y_lower[0]]

            # Desenhar as aletas (superior e inferior)
            ax.plot(fin_x_upper, fin_y_upper_closed, color="blue")
            ax.plot(fin_x_lower, fin_y_lower_closed, color="blue")

        # Configurações do plot
        ax.set_aspect('equal', adjustable='datalim')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel("x, m")
        ax.set_ylabel("y, m")
        ax.set_title("Missile Geometry")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()


# Exemplo de uso
input_file = "for005.dat"  # Nome do arquivo gerado pelo MissileDATCOMInput
plotter = MissileGeometryPlotterFromFile(input_file)
plotter.parse_input_file()
plotter.plot_geometry()
