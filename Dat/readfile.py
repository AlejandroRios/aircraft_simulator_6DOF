import re
import os
import pandas as pd

class MissileDATCOMOutput:
    def __init__(self, filepath='for006.dat'):
        self.filepath = filepath
        self.results = {}
        self.block_counter = {}

        if not filepath:
            raise ValueError('A rota do arquivo de entrada não foi encontrada')

    def _card_names(self):
        return [
            'AXISYMMETRIC BODY DEFINITION', 'MOLD LINE CONTOUR', 'FIN SET NUMBER 1 AIRFOIL SECTION',
            'GEOMETRIC RESULTS FOR FIN SETS', 'FIN SET SECTION AIRFOIL', 'PROTUBERANCE OUTPUT',
            'BODY ALONE PARTIAL OUTPUT', 'FIN SET CA PARTIAL OUTPUT', 'FIN SET CN, CM PARTIAL OUTPUT',
            'AERODYNAMIC FORCE AND MOMENT SYNTHESIS', 'FIN SET PANEL BENDING MOMENTS (ABOUT EXPOSED ROOT CHORD)',
            'FIN SET PANEL HINGE MOMENTS (ABOUT HINGE LINE)', 'STATIC AERODYNAMICS FOR BODY-FIN SET',
            'BODY ALONE DYNAMIC DERIVATIVES', 'BODY + FIN SET DYNAMIC DERIVATIVES'
        ]

    def read_input_file(self):
        with open(self.filepath, 'r') as f:
            file_lines = f.readlines()

        blocks = self._identify_blocks(file_lines)
        self._store_results(blocks)
        self._export_to_csv()
        self._print_full_results()

    def _identify_blocks(self, file_lines):
        """
        Identify blocks in the input file and handle multiple occurrences.
        """
        blocks = []
        current_header = None
        current_block = []
        header_counts = {}  # To track occurrences of each block

        cards = self._card_names()
        detected_headers = []  # List to store detected headers for printing

        for line in file_lines:
            line = line.strip()
            # Detect specific patterns for BODY ALONE and BODY + FIN SET
            if "BODY ALONE DYNAMIC DERIVATIVES" in line.upper():
                header = "BODY ALONE DYNAMIC DERIVATIVES"
            elif "BODY + 1 FIN SET DYNAMIC DERIVATIVES" in line.upper():
                header = "BODY + FIN SET DYNAMIC DERIVATIVES"
            elif line in cards:
                header = line.strip()
            else:
                header = None

            if header:
                # Store the previous block
                if current_header is not None:
                    block_id = f"{current_header}_{header_counts[current_header]}"
                    blocks.append((block_id, current_block))

                # Start a new block
                current_header = header
                header_counts[current_header] = header_counts.get(current_header, 0) + 1
                current_block = []

                # Add header to the list of detected headers
                detected_headers.append(header)

            else:
                current_block.append(line.strip())

        # Add the last block
        if current_header is not None:
            block_id = f"{current_header}_{header_counts[current_header]}"
            blocks.append((block_id, current_block))

        # Print the detected blocks
        print("\nBlocos detectados:")
        for header in detected_headers:
            print(f" - {header}")

        print(f"\nNúmero total de blocos detectados: {len(detected_headers)}")
        return blocks


    def _generate_short_name(self, header):
        """
        Gera um nome curto único para o bloco com base no cabeçalho e na contagem de ocorrências.
        """
        header = header.upper()
        if "BODY ALONE DYNAMIC DERIVATIVES" in header:
            base_name = "body_alone_dynamic"
        elif "BODY + FIN SET DYNAMIC DERIVATIVES" in header:
            base_name = "body_plus_fin_set_dynamic"
        else:
            base_name = "_".join(header.lower().split()[:3])

        count = self.block_counter.get(base_name, 0) + 1
        self.block_counter[base_name] = count
        return f"{base_name}_{count}"

    def _store_results(self, blocks):
        for block_id, block in blocks:
            header, occurrence = block_id.rsplit('_', 1)
            occurrence = int(occurrence)

            # Processar o bloco
            tables, data = self._table_converter(block, header)

            # Debugging: Mostrar as tabelas e os dados processados
            print(f"\nProcessing Block: {block_id}")
            print(f"Tables: {tables}")
            print(f"Data: {data}")

            # Verificar se o bloco contém dados relevantes
            if not tables and not data:
                print(f"Skipping empty block: {block_id}")
                continue

            # Estruturar os dados e armazená-los
            structured_data = self._structure_data(header, tables, data)
            short_name = self._generate_short_name(block_id)
            self.results[short_name] = structured_data


    def _table_converter(self, block, header):
        """
        Processa o bloco baseado no cabeçalho e retorna tabelas e dados.
        """
        if header == 'AXISYMMETRIC BODY DEFINITION':
            return self._process_axisymmetric_body_definition(block), []
        elif header == 'MOLD LINE CONTOUR':
            return self._process_mold_line_contour(block), []
        elif header == 'GEOMETRIC RESULTS FOR FIN SETS':
            return self._process_geometric_results_for_fin_sets(block), []
        elif header == 'PROTUBERANCE OUTPUT':
            return self._process_protuberance_output(block), []
        elif header == 'FIN SET NUMBER 1 AIRFOIL SECTION':
            return self._process_fin_set_number_1_airfoil_section(block), []
        elif header == 'BODY ALONE PARTIAL OUTPUT':
            return self._process_body_alone_partial_output(block), []
        elif header == 'BODY ALONE DYNAMIC DERIVATIVES':
            return self._process_body_alone_dynamic_derivatives(block), []
        elif header == 'BODY + FIN SET DYNAMIC DERIVATIVES':
            return self._process_body_plus_fin_set_dynamic_derivatives(block), []
        elif header == 'FIN SET CA PARTIAL OUTPUT':
            return self._process_fin_set_ca_partial_output(block), []
        elif header == 'FIN SET CN, CM PARTIAL OUTPUT':
            return self._process_fin_set_cn_cm_partial_output(block), []
        elif header == 'AERODYNAMIC FORCE AND MOMENT SYNTHESIS':
            return self._process_aerodynamic_force_and_moment_synthesis(block), []
        elif header == 'STATIC AERODYNAMICS FOR BODY-FIN SET':
            return self._process_static_aerodynamics_for_body_fin_set(block), []
        elif header == 'STATIC AERODYNAMICS FOR BODY-FIN SET 1':
            if "DERIVATIVES" in block[0]:
                return self._process_static_aerodynamics_derivatives(block), []
            else:
                return self._process_static_aerodynamics_for_body_fin_set(block), []
        else:
            return self._process_generic_table(block), []


    def _process_axisymmetric_body_definition(self, block):
        """
        Processa o bloco 'AXISYMMETRIC BODY DEFINITION'.
        """
        headers = ["ITEM", "NOSE", "CENTERBODY", "AFT BODY", "TOTAL", "UNIT"]
        data_rows = []

        for line in block:
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) >= 5:
                item = parts[0]
                nose, centerbody, aft_body, total = parts[1:5]
                unit = parts[5] if len(parts) > 5 else ""
                data_rows.append([item, nose, centerbody, aft_body, total, unit])

        table_df = pd.DataFrame(data_rows, columns=headers)
        return [{'header': headers, 'data': table_df}], []

    def _process_mold_line_contour(self, block):
        """
        Processa o bloco 'MOLD LINE CONTOUR'.
        """
        # Processar o bloco específico para MOLD LINE CONTOUR
        longitudinal_stations = []
        body_radii = []
        processing_list = None

        for line in block:
            line = line.strip()
            if line.startswith("LONGITUDINAL STATIONS"):
                processing_list = "LONGITUDINAL STATIONS"
                # Capturar valores na mesma linha, após o título
                values = line[len("LONGITUDINAL STATIONS"):].strip().split()
                longitudinal_stations.extend(values)
                continue
            elif line.startswith("BODY RADII"):
                processing_list = "BODY RADII"
                # Capturar valores na mesma linha, após o título
                values = line[len("BODY RADII"):].strip().split()
                body_radii.extend(values)
                continue

            # Detectar o final do bloco baseado em texto fora do padrão
            if line.startswith("NOTE") or line.startswith("*****"):
                break  # Final do bloco detectado

            # Processar linhas subsequentes de valores
            if processing_list == "LONGITUDINAL STATIONS" and line:
                longitudinal_stations.extend(line.split())
            elif processing_list == "BODY RADII" and line:
                body_radii.extend(line.split())

        # Convertendo listas para floats e criando o DataFrame

        longitudinal_stations = [float(x.replace("*", "")) for x in longitudinal_stations]
        body_radii = [float(x.replace("*", "")) for x in body_radii]
        mold_line_df = pd.DataFrame({
            "LONGITUDINAL STATIONS": longitudinal_stations,
            "BODY RADII": body_radii
        })

        return [{'header': ["LONGITUDINAL STATIONS", "BODY RADII"], 'data': mold_line_df}], []


    def _process_geometric_results_for_fin_sets(self, block):
        """
        Processa o bloco 'GEOMETRIC RESULTS FOR FIN SETS'.
        Extrai a tabela contendo as colunas SEGMENT PLAN AREA e suas linhas.
        """
        headers = [
            "SEGMENT NUMBER", "PLAN AREA", "ASPECT RATIO", "TAPER RATIO",
            "L.E. SWEEP", "T.E. SWEEP", "M.A.C. CHORD", "T/C RATIO"
        ]
        data_rows = []
        processing_data = False

        for line in block:
            line = line.strip()

            # Ignorar linhas irrelevantes ou em branco
            if not line or "*****" in line or "PAGE" in line or "MK82" in line or "DATA FOR ONE PANEL ONLY" in line:
                continue

            # Detectar início do bloco de dados
            if "SEGMENT" in line and "PLAN" in line:
                processing_data = True
                continue

            # Processar linhas de dados
            if processing_data:
                # Linhas válidas de dados
                if line.startswith("TOTAL") or line[0].isdigit():
                    data = re.split(r'\s{2,}', line.strip())  # Dividir por múltiplos espaços
                    data_rows.append(data)

        # Criar DataFrame se os dados forem encontrados
        if data_rows:
            df = pd.DataFrame(data_rows, columns=headers)
            return [{'header': headers, 'data': df}], []

        return [], []


    def _process_protuberance_output(self, block):
        """
        Processa o bloco 'PROTUBERANCE OUTPUT'.
        Extrai a tabela contendo os dados das protuberâncias.
        """
        headers = ["NUMBER", "TYPE", "LONG. LOCATION (M)", "NUMBER", "INDIVIDUAL CA", "TOTAL CA"]
        data_rows = []
        processing_data = False

        for line in block:
            line = line.strip()

            # Ignorar linhas irrelevantes
            if not line or "*****" in line or "PAGE" in line or "MK82" in line:
                continue

            # Detectar início do bloco de dados
            if "--------------- PROTUBERANCE  CALCULATIONS ---------------" in line:
                processing_data = True
                continue

            # Parar processamento no final do bloco
            if "TOTAL CA DUE TO PROTUBERANCES" in line:
                break

            # Processar linhas de dados
            if processing_data and line[0].isdigit():
                data = re.split(r'\s{2,}', line.strip())  # Dividir por múltiplos espaços
                data_rows.append(data)

        # Criar DataFrame se os dados forem encontrados
        if data_rows:
            df = pd.DataFrame(data_rows, columns=headers)
            return [{'header': headers, 'data': df}], []

        return [], []


    def _process_fin_set_number_1_airfoil_section(self, block):
        """
        Processa o bloco 'FIN SET NUMBER 1 AIRFOIL SECTION'.
        """
        headers = None
        data_rows = []
        processing_data = False

        for line in block:
            line = line.strip()

            # Ignorar linhas irrelevantes
            if not line or "*****" in line or "AERODYNAMIC METHODS" in line or "PAGE" in line:
                continue

            # Detectar cabeçalhos
            if not processing_data and "X/C" in line:
                headers = re.split(r'\s{2,}', line.strip())
                processing_data = True
                continue

            # Processar linhas de dados
            if processing_data:
                data = re.split(r'\s+', line.strip())
                if headers and len(data) == len(headers):  # Garantir que os dados correspondem aos cabeçalhos
                    data_rows.append(data)

        # Criar DataFrame se os dados forem encontrados
        if headers and data_rows:
            df = pd.DataFrame(data_rows, columns=headers)
            return [{'header': headers, 'data': df}], []

        return [], []


    def _process_fin_set_section_airfoil(self, block):
        """
        Processa o bloco 'FIN SET SECTION AIRFOIL'.
        """
        headers = None
        data_rows = []
        for line in block:
            line = line.strip()
            if not line:
                continue
            if headers is None and "X/C" in line:
                headers = re.split(r'\s{2,}', line.strip())
                continue
            if headers:
                data = re.split(r'\s+', line.strip())
                if len(data) == len(headers):
                    data_rows.append(data)
        if headers and data_rows:
            df = pd.DataFrame(data_rows, columns=headers)
            return [{'header': headers, 'data': df}], []
        return [], []

    def _process_body_alone_partial_output(self, block):
        """
        Processa o bloco 'BODY ALONE PARTIAL OUTPUT' e separa em duas tabelas.
        """
        table_1 = self._process_body_alone_table_1(block)
        table_2 = self._process_body_alone_table_2(block)
        return [table_1, table_2], []

    def _process_body_alone_table_1(self, block):
        """
        Processa a primeira tabela do bloco 'BODY ALONE PARTIAL OUTPUT'.
        Colunas: ALPHA, CA-FRIC, CA-PRES/WAVE, CA-BASE, CA-PROT, CA-SEP, CA-ALP.
        """
        headers = None
        data_rows = []
        processing_data = False

        for line in block:
            line = line.strip()

            # Detectar cabeçalho da tabela 1
            if not processing_data and "ALPHA    CA-FRIC" in line:
                headers = re.split(r'\s{2,}', line.strip())
                processing_data = True
                continue

            # Processar linhas de dados
            if processing_data:
                if not line or "CROSS FLOW DRAG PROPORTIONALITY FACTOR" in line:
                    break  # Final da tabela
                data = re.split(r'\s+', line.strip())
                if len(data) == len(headers):
                    data_rows.append(data)

        # Criar DataFrame
        if headers and data_rows:
            return {'header': headers, 'data': pd.DataFrame(data_rows, columns=headers)}

        return {'header': [], 'data': pd.DataFrame()}

    def _process_body_alone_table_2(self, block):
        """
        Processa a segunda tabela do bloco 'BODY ALONE PARTIAL OUTPUT'.
        Colunas: ALPHA, CN-POTEN, CN-VISC, CN-SEP, CM-POTEN, CM-VISC, CM-SEP, CDC.
        """
        headers = None
        data_rows = []
        processing_data = False

        for line in block:
            line = line.strip()

            # Detectar cabeçalho da tabela 2
            if not processing_data and "ALPHA    CN-POTEN" in line:
                headers = re.split(r'\s{2,}', line.strip())  # Dividir por múltiplos espaços
                processing_data = True
                continue

            # Processar linhas de dados da tabela
            if processing_data:
                # Parar o processamento ao encontrar um marcador de fim de tabela
                if not line or "*****" in line or "PAGE" in line:
                    break

                # Dividir a linha com cuidado para preservar colunas vazias
                data = re.split(r'\s+', line.strip())

                # Preencher colunas ausentes com `None`
                while len(data) < len(headers):
                    data.append(None)

                # Validar o comprimento e processar os dados
                if len(data) == len(headers):
                    data_rows.append(data)

        # Criar DataFrame se os dados forem encontrados
        if headers and data_rows:
            df = pd.DataFrame(data_rows, columns=headers)

            # Converter valores numéricos e manter `None` para células vazias
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return {'header': headers, 'data': df}

        # Caso não seja encontrado nenhum dado, retornar uma tabela vazia
        return {'header': [], 'data': pd.DataFrame()}

    def _process_fin_set_ca_partial_output(self, block):
        """
        Processa o bloco 'FIN SET CA PARTIAL OUTPUT'.
        Retorna valores escalares e a tabela de força axial devido ao ângulo de ataque.
        """
        scalar_values = {}
        headers = None
        data_rows = []
        processing_table = False

        for line in block:
            line = line.strip()

            # Ignorar linhas irrelevantes
            if not line or "*****" in line or "PAGE" in line or "MK82" in line:
                continue

            # Detectar e processar valores escalares
            if "SINGLE FIN PANEL ZERO-LIFT AXIAL FORCE COMPONENTS" in line:
                processing_table = False  # Encerrar qualquer tabela em processamento
                continue
            if "FIN AXIAL FORCE DUE TO ANGLE OF ATTACK" in line:
                processing_table = True  # Iniciar processamento da tabela
                continue

            if not processing_table:
                # Processar valores escalares
                if "TOTAL CAO" in line or any(keyword in line for keyword in ["SKIN", "SUBSONIC", "TRANSONIC", "SUPERSONIC", "LEADING", "TRAILING"]):
                    parts = line.split()
                    key = " ".join(parts[:-1]).strip()
                    value = parts[-1].strip()
                    try:
                        scalar_values[key] = float(value)
                    except ValueError:
                        scalar_values[key] = value

            if processing_table:
                # Detectar cabeçalhos da tabela
                if headers is None and "ALPHA" in line:
                    headers = re.split(r'\s{2,}', line.strip())
                    continue

                # Processar linhas de dados da tabela
                if headers:
                    data = re.split(r'\s+', line.strip())
                    if len(data) == len(headers):
                        data_rows.append(data)

        # Criar DataFrame para a tabela
        table_df = pd.DataFrame(data_rows, columns=headers) if data_rows else pd.DataFrame()

        # Converter valores numéricos no DataFrame
        for col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors='coerce')

        # Retornar valores escalares e tabela
        return {'scalar_values': scalar_values, 'table': table_df}

    def _process_fin_set_cn_cm_partial_output(self, block):
        """
        Processa o bloco 'FIN SET CN, CM PARTIAL OUTPUT'.
        Retorna:
        - Valores escalares encontrados no bloco (e.g., NORMAL FORCE SLOPE, CENTER OF PRESSURE).
        - DataFrame contendo a tabela CN, CM (ALPHA, CN LINEAR, NON-LINEAR, etc.).
        """
        scalar_values = {}
        headers = None
        data_rows = []
        processing_table = False

        for line in block:
            line = line.strip()

            # Ignorar linhas irrelevantes
            if not line or "*****" in line or "PAGE" in line or "MK82" in line:
                continue

            # Detectar valores escalares
            if "NORMAL FORCE SLOPE" in line or "CENTER OF PRESSURE" in line:
                parts = line.split("=")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().split()[0]  # Apenas o valor numérico
                    try:
                        scalar_values[key] = float(value)
                    except ValueError:
                        scalar_values[key] = value
                continue

            # Detectar início da tabela
            if "ALPHA" in line and "CN" in line and "CM" in line:
                headers = re.split(r'\s{2,}', line.strip())  # Dividir por múltiplos espaços
                processing_table = True
                continue

            # Processar linhas da tabela
            if processing_table:
                # Detectar o fim da tabela
                if "*****" in line or not line:
                    break

                data = re.split(r'\s+', line.strip())
                if len(data) == len(headers):  # Garantir que o número de colunas corresponde
                    data_rows.append(data)

        # Criar DataFrame da tabela, se encontrado
        table_df = pd.DataFrame(data_rows, columns=headers) if data_rows else pd.DataFrame()

        # Converter colunas numéricas no DataFrame
        for col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors='coerce')

        return {'scalar_values': scalar_values, 'table': table_df}

    def _process_aerodynamic_force_and_moment_synthesis(self, block):
        """
        Processa o bloco 'AERODYNAMIC FORCE AND MOMENT SYNTHESIS'.
        Extrai a tabela com as colunas ALPHA, CN, CM, CA, CY, CLN, CLL.
        """
        headers = None
        data_rows = []
        processing_table = False

        for line in block:
            line = line.strip()

            # Ignorar linhas irrelevantes
            if not line or "*****" in line or "PAGE" in line or "MK82" in line:
                continue

            # Detectar cabeçalho da tabela
            if "ALPHA" in line and "CN" in line and "CLL" in line:
                headers = re.split(r'\s{2,}', line.strip())  # Dividir por múltiplos espaços
                processing_table = True
                continue

            # Processar linhas da tabela
            if processing_table:
                # Detectar o fim da tabela
                if "*****" in line or not line:
                    break

                data = re.split(r'\s+', line.strip())
                if len(data) == len(headers):  # Garantir que o número de colunas corresponde
                    data_rows.append(data)

        # Criar DataFrame da tabela, se encontrado
        table_df = pd.DataFrame(data_rows, columns=headers) if data_rows else pd.DataFrame()

        # Converter colunas numéricas no DataFrame
        for col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors='coerce')

        return {'header': headers, 'data': table_df}

    def _process_aerodynamic_force_and_moment_synthesis(self, block):
        """
        Processa o bloco 'AERODYNAMIC FORCE AND MOMENT SYNTHESIS'.
        """
        headers = ["ALPHA", "CN", "CM", "CA", "CY", "CLN", "CLL"]
        data_rows = []

        for line in block:
            line = line.strip()

            # Ignorar linhas irrelevantes
            if not line or "*****" in line or "PAGE" in line:
                continue

            # Processar linhas de dados
            if line[0].isdigit() or line[0] == "-":  # Detecta se a linha começa com um número
                data = re.split(r'\s+', line.strip())  # Divide por espaços múltiplos
                if len(data) == len(headers):
                    data_rows.append(data)
                else:
                    print(f"Aviso: Número inesperado de colunas em linha: {line}")
                    print(f"Dados encontrados: {data}")
                    print(f"Esperado: {len(headers)} colunas")

        # Criar DataFrame da tabela, se encontrado
        table_df = pd.DataFrame(data_rows, columns=headers) if data_rows else pd.DataFrame()

        # Converter colunas numéricas no DataFrame
        for col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors='coerce')

        return {'header': headers, 'data': table_df}

    def _process_static_aerodynamics_body_fin_set(self, block):
        """
        Processa o bloco 'STATIC AERODYNAMICS FOR BODY-FIN SET 1'.
        Junta as duas tabelas em uma única estrutura.
        """
        headers_table1 = None
        headers_table2 = None
        data_rows_table1 = []
        data_rows_table2 = []
        processing_table = 1  # Identifica qual tabela está sendo processada

        for line in block:
            line = line.strip()

            # Ignorar linhas irrelevantes
            if not line or "*****" in line or "PAGE" in line or "MK82" in line:
                continue

            # Detectar cabeçalho da tabela 1
            if processing_table == 1 and "ALPHA" in line and "CN" in line and "CM" in line:
                headers_table1 = re.split(r'\s{2,}', line.strip())  # Dividir por múltiplos espaços
                continue

            # Detectar cabeçalho da tabela 2
            if processing_table == 1 and "ALPHA" in line and "CL" in line and "CD" in line:
                # Salvar a tabela 1 antes de mudar
                processing_table = 2
                headers_table2 = re.split(r'\s{2,}', line.strip())
                continue

            # Processar linhas da tabela 1
            if processing_table == 1 and headers_table1:
                data = re.split(r'\s+', line.strip())
                if len(data) == len(headers_table1):  # Validar número de colunas
                    data_rows_table1.append(data)

            # Processar linhas da tabela 2
            if processing_table == 2 and headers_table2:
                data = re.split(r'\s+', line.strip())
                if len(data) == len(headers_table2):  # Validar número de colunas
                    data_rows_table2.append(data)

        # Criar DataFrames
        table1_df = pd.DataFrame(data_rows_table1, columns=headers_table1) if data_rows_table1 else pd.DataFrame()
        table2_df = pd.DataFrame(data_rows_table2, columns=headers_table2) if data_rows_table2 else pd.DataFrame()

        # Converter colunas numéricas
        for col in table1_df.columns:
            table1_df[col] = pd.to_numeric(table1_df[col], errors='coerce')
        for col in table2_df.columns:
            table2_df[col] = pd.to_numeric(table2_df[col], errors='coerce')

        # Combinar tabelas usando a coluna "ALPHA" como referência
        combined_df = pd.merge(table1_df, table2_df, on="ALPHA", how="inner")

        return {'header': combined_df.columns.tolist(), 'data': combined_df}

    def _process_static_aerodynamics_derivatives(self, block):
        """
        Processa o bloco 'STATIC AERODYNAMICS FOR BODY-FIN SET 1'.
        Extrai os dados das tabelas de Derivativos e Ângulos de Deflexão dos Paineis.
        """
        # Inicialização
        headers_derivatives = None
        data_rows_derivatives = []
        panel_headers = None
        panel_data = []
        processing_table = "DERIVATIVES"  # Alterna entre as tabelas

        for line in block:
            line = line.strip()

            # Ignorar linhas irrelevantes
            if not line or "*****" in line or "PAGE" in line or "MK82" in line:
                continue

            # Detectar cabeçalho da tabela de Derivativos
            if processing_table == "DERIVATIVES" and "ALPHA" in line and "CNA" in line:
                headers_derivatives = re.split(r'\s{2,}', line.strip())
                continue

            # Detectar tabela de Ângulos de Deflexão
            if processing_table == "DERIVATIVES" and "PANEL DEFLECTION ANGLES" in line:
                processing_table = "PANEL_ANGLES"
                continue

            # Processar dados da tabela de Derivativos
            if processing_table == "DERIVATIVES" and headers_derivatives:
                data = re.split(r'\s+', line.strip())
                if len(data) == len(headers_derivatives):  # Validar número de colunas
                    data_rows_derivatives.append(data)

            # Detectar cabeçalhos da tabela de Ângulos de Deflexão
            if processing_table == "PANEL_ANGLES" and "SET" in line and "FIN 1" in line:
                panel_headers = re.split(r'\s{2,}', line.strip())
                continue

            # Processar dados da tabela de Ângulos de Deflexão
            if processing_table == "PANEL_ANGLES" and panel_headers:
                data = re.split(r'\s+', line.strip())
                if len(data) == len(panel_headers):  # Validar número de colunas
                    panel_data.append(data)

        # Criar DataFrames
        derivatives_df = (
            pd.DataFrame(data_rows_derivatives, columns=headers_derivatives)
            if data_rows_derivatives
            else pd.DataFrame()
        )
        panel_df = (
            pd.DataFrame(panel_data, columns=panel_headers)
            if panel_data
            else pd.DataFrame()
        )

        # Converter colunas numéricas
        for col in derivatives_df.columns:
            derivatives_df[col] = pd.to_numeric(derivatives_df[col], errors="coerce")
        for col in panel_df.columns:
            panel_df[col] = pd.to_numeric(panel_df[col], errors="coerce")

        return {
            "derivatives": {"header": derivatives_df.columns.tolist(), "data": derivatives_df},
            "panel_deflection_angles": {"header": panel_df.columns.tolist(), "data": panel_df},
        }

    def _process_static_aerodynamics_for_body_fin_set(self, block):
        """
        Processa o bloco 'STATIC AERODYNAMICS FOR BODY-FIN SET'.
        Identifica e extrai múltiplas tabelas dentro do bloco.
        """
        data_sets = []
        headers = None
        data_rows = []
        processing_data = False

        for line in block:
            line = line.strip()

            # Ignorar linhas irrelevantes
            if not line or "*****" in line or "PAGE" in line:
                continue

            # Detectar o início de tabelas específicas
            if any(keyword in line for keyword in ["ALPHA", "CNA", "CL/CD", "DERIVATIVES"]):
                # Salvar tabela anterior
                if headers and data_rows:
                    df = pd.DataFrame(data_rows, columns=headers)
                    data_sets.append({'header': headers, 'data': df})
                    data_rows = []  # Reiniciar para nova tabela

                # Detectar cabeçalhos da nova tabela
                headers = re.split(r'\s+', line.strip())
                processing_data = True
                continue

            # Processar linhas de dados
            if processing_data:
                data = re.split(r'\s+', line.strip())
                # Ignorar linhas que não têm o mesmo número de colunas que os cabeçalhos
                if headers and len(data) == len(headers):
                    try:
                        # Converter valores numéricos para float
                        data = [float(x) if x.replace('.', '', 1).replace('-', '', 1).isdigit() else x for x in data]
                        data_rows.append(data)
                    except ValueError:
                        continue
                else:
                    # Final da tabela detectado
                    processing_data = False

        # Salvar a última tabela, se presente
        if headers and data_rows:
            df = pd.DataFrame(data_rows, columns=headers)
            data_sets.append({'header': headers, 'data': df})

        # Retornar as tabelas processadas
        return data_sets, []

    def _process_body_alone_dynamic_derivatives(self, block):
        """
        Processa o bloco 'BODY ALONE DYNAMIC DERIVATIVES'.
        Permite lidar com múltiplos conjuntos de dados no mesmo bloco.
        """
        data_sets = []
        headers = None
        data_rows = []
        processing_data = False

        for line in block:
            line = line.strip()

            # Ignorar linhas irrelevantes
            if not line or "*****" in line or "PAGE" in line or "FLIGHT CONDITIONS" in line:
                continue

            # Detectar cabeçalhos e reiniciar o processamento de dados
            if "ALPHA" in line and any(key in line for key in ["CNQ", "CYR", "CMQ", "CLNR", "CLLP"]):
                # Salvar o conjunto de dados anterior (se existir)
                if headers and data_rows:
                    df = pd.DataFrame(data_rows, columns=headers)
                    data_sets.append({'header': headers, 'data': df})
                    data_rows = []  # Reiniciar para o próximo conjunto de dados

                headers = re.split(r'\s+', line.strip())  # Detectar cabeçalhos separados por espaços
                processing_data = True
                continue

            # Processar linhas de dados
            if processing_data:
                data = re.split(r'\s+', line.strip())
                # Ignorar linhas que não têm o mesmo número de colunas que os cabeçalhos
                if headers and len(data) == len(headers):
                    try:
                        # Converter valores numéricos para float
                        data = [float(x) if x.replace('.', '', 1).replace('-', '', 1).isdigit() else x for x in data]
                        data_rows.append(data)
                    except ValueError:
                        continue
                else:
                    # Se encontrar algo incompatível, parar de processar dados
                    processing_data = False

            # Parar o processamento ao encontrar linhas que claramente não são dados
            if "DERIVATIVES" in line.upper() and not line.startswith("ALPHA"):
                processing_data = False

        # Salvar o último conjunto de dados
        if headers and data_rows:
            # Remover qualquer linha desnecessária ao final
            if data_rows[-1][0] == "PITCH" or data_rows[-1][0] == "YAW":
                data_rows.pop(-1)
            df = pd.DataFrame(data_rows, columns=headers)
            data_sets.append({'header': headers, 'data': df})

        # Retornar os conjuntos de dados processados
        return data_sets, []



    def _process_body_plus_fin_set_dynamic_derivatives(self, block):
        """
        Processa o bloco 'BODY + FIN SET DYNAMIC DERIVATIVES'.
        Permite lidar com múltiplos conjuntos de dados no mesmo bloco.
        """
        data_sets = []
        headers = None
        data_rows = []
        processing_data = False

        for line in block:
            line = line.strip()

            # Ignorar linhas irrelevantes
            if not line or "*****" in line or "PAGE" in line or "FLIGHT CONDITIONS" in line:
                continue

            # Detectar cabeçalhos e reiniciar o processamento de dados
            if "ALPHA" in line and any(key in line for key in ["CNQ", "CYR", "CMQ", "CLNR", "CLLP"]):
                # Salvar o conjunto de dados anterior (se existir)
                if headers and data_rows:
                    df = pd.DataFrame(data_rows, columns=headers)
                    data_sets.append({'header': headers, 'data': df})
                    data_rows = []  # Reiniciar para o próximo conjunto de dados

                headers = re.split(r'\s+', line.strip())  # Detectar cabeçalhos separados por espaços
                processing_data = True
                continue

            # Processar linhas de dados
            if processing_data:
                data = re.split(r'\s+', line.strip())
                # Ignorar linhas que não têm o mesmo número de colunas que os cabeçalhos
                if headers and len(data) == len(headers):
                    try:
                        # Converter valores numéricos para float
                        data = [float(x) if x.replace('.', '', 1).replace('-', '', 1).isdigit() else x for x in data]
                        data_rows.append(data)
                    except ValueError:
                        continue
                else:
                    # Se encontrar algo incompatível, parar de processar dados
                    processing_data = False

            # Parar o processamento ao encontrar linhas que claramente não são dados
            if "DERIVATIVES" in line.upper() and not line.startswith("ALPHA"):
                processing_data = False

        # Salvar o último conjunto de dados
        if headers and data_rows:
            # Remover qualquer linha desnecessária ao final
            if data_rows[-1][0] == "PITCH" or data_rows[-1][0] == "YAW":
                data_rows.pop(-1)
            df = pd.DataFrame(data_rows, columns=headers)
            data_sets.append({'header': headers, 'data': df})

        # Retornar os conjuntos de dados processados
        return data_sets, []




    def _process_generic_table(self, block):
        """
        Processa tabelas genéricas detectando automaticamente cabeçalhos e dados numéricos.
        Ajusta cabeçalhos com espaçamentos fixos e palavras compostas.
        """
        headers = None
        data_rows = []

        for line in block:
            # Ignorar linhas irrelevantes (e.g., "NACA S-3-00.5-01.7-99.0")
            if not line.strip() or re.match(r"^[A-Za-z\-]+\s+[A-Za-z\-]+", line):
                continue

            # Detectar o cabeçalho: fixar as colunas com base no espaçamento
            if headers is None and "X/C" in line:
                headers = re.split(r'\s{2,}', line.strip())  # Divisão por múltiplos espaços
                continue

            # Adicionar linhas numéricas à tabela
            if headers:
                data = re.split(r'\s+', line.strip())
                if len(data) == len(headers):  # Garantir correspondência de colunas
                    try:
                        # Converter valores em floats
                        data = [float(value) for value in data]
                        data_rows.append(data)
                    except ValueError:
                        # Ignorar linhas que não são numéricas
                        continue

        # Criar DataFrame se os dados forem encontrados
        if headers and data_rows:
            return [{'header': headers, 'data': pd.DataFrame(data_rows, columns=headers)}]

        return []


    def _structure_data(self, header, tables, data):
        structured_data = {
            'tables': [],
            'lists': [],
            'values': {}
        }

        # Add tables to the structured data
        for table in tables:
            if not isinstance(table, dict) or 'data' not in table:
                print(f"Invalid table format for header '{header}': {table}")
                continue

            try:
                df = table['data']
                structured_data['tables'].append(df)
            except ValueError as e:
                print(f"Error creating DataFrame for header '{header}': {e}")
                print(f"Headers: {table.get('header', [])}")
                print(f"Data: {table.get('data', [])}")

        # Add non-tabular data
        for line in data:
            if '=' in line:
                key, value = map(str.strip, line.split('=', 1))
                try:
                    value = float(value)
                except ValueError:
                    pass
                structured_data['values'][key] = value
            else:
                structured_data['lists'].append(line)

        return structured_data



    def _export_to_csv(self):
        output_dir = "output/extracted_tables/"
        os.makedirs(output_dir, exist_ok=True)
        for header, content in self.results.items():
            for idx, table in enumerate(content['tables']):
                filename = f"{output_dir}/{header}_table_{idx + 1}.csv"
                table.to_csv(filename, index=False)
                print(f"Tabela extraída salva em: {filename}")

    def _print_full_results(self):
        print("\nDicionário Completo de Resultados:")
        for key, value in self.results.items():
            print(f"\nHeader: {key}")
            print("Tables:")
            for idx, table in enumerate(value['tables']):
                print(f"Table {idx + 1}:\n{table}")
            print("Values:")
            for k, v in value['values'].items():
                print(f"{k}: {v}")
            print("Lists:")
            print(value['lists'])

# Criar instância e ler o arquivo de entrada
input_padrao = MissileDATCOMOutput('for006.dat')
input_padrao.read_input_file()


