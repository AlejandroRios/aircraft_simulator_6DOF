import re
import pandas as pd
import io
import os

# ------------------------------
# Função 1: Identificar Blocos
# ------------------------------
def identify_blocks(file_lines, header_pattern):
    """
    Identifica e organiza blocos no arquivo com base em cabeçalhos.
    
    Parameters:
        file_lines (list): Linhas do arquivo.
        header_pattern (str): Regex para identificar cabeçalhos.

    Returns:
        list: Lista de tuplas (cabeçalho, bloco de linhas).
    """
    pattern = re.compile(header_pattern)
    blocks = []
    current_header = None
    current_block = []

    for line in file_lines:
        if pattern.match(line):  # Detecta cabeçalhos
            if current_header is not None:
                blocks.append((current_header, current_block))
            current_header = line.strip()
            current_block = []  # Começa um novo bloco
        else:
            current_block.append(line.strip())

    # Salvar o último bloco, se existir
    if current_header is not None:
        blocks.append((current_header, current_block))
    
    return blocks

# ------------------------------
# Função 2: Processar Tabelas
# ------------------------------
def extract_tables_general(block):
    """
    Extrai tabelas de um bloco, detectando diferentes estruturas.
    
    Parameters:
        block (list): Bloco de linhas de texto.

    Returns:
        list: Lista de DataFrames representando tabelas extraídas.
    """
    tables = []
    header_lines = []
    data_lines = []

    for line in block:
        # Detectar cabeçalhos multi-linha
        if re.match(r"^[A-Za-z\s\(\)\-]+$", line) and not re.search(r"\d", line):
            header_lines.append(line.strip())
        # Detectar linhas numéricas para dados
        elif re.match(r"^\s*[\d\.\-\+\sEe]+$", line):
            data_lines.append(line.strip())
        else:
            # Processar tabela ao encontrar uma quebra de padrão
            if data_lines:
                table = process_table(header_lines, data_lines)
                if table is not None:
                    tables.append(table)
                header_lines = []
                data_lines = []

    # Processar última tabela no bloco
    if data_lines:
        table = process_table(header_lines, data_lines)
        if table is not None:
            tables.append(table)

    return tables

def process_table(header_lines, data_lines):
    """
    Processa e converte cabeçalhos e dados em DataFrame.
    
    Parameters:
        header_lines (list): Linhas de cabeçalho.
        data_lines (list): Linhas de dados.

    Returns:
        DataFrame: Tabela convertida ou None se falhar.
    """
    try:
        # Diagnóstico: Detectar o número máximo de colunas
        max_columns = max(len(line.split()) for line in data_lines)
        print(f"Detectado número máximo de colunas: {max_columns}")

        # Ajustar todas as linhas de dados para o número máximo de colunas
        adjusted_data_lines = []
        for line in data_lines:
            split_line = line.split()
            if len(split_line) < max_columns:
                split_line.extend([''] * (max_columns - len(split_line)))  # Completar com vazios
            adjusted_data_lines.append(" ".join(split_line))

        # Combinar cabeçalho e linhas ajustadas
        combined_lines = header_lines + adjusted_data_lines
        table_text = "\n".join(combined_lines)

        # Tentar processar como CSV com separadores flexíveis
        table = pd.read_csv(io.StringIO(table_text), sep=r'\s+')

        # Se não houver cabeçalho detectado, criar colunas genéricas
        if table.columns[0] == 0:
            table.columns = [f"Column_{i+1}" for i in range(len(table.columns))]

        return table

    except Exception as e:
        print(f"Erro ao processar tabela: {e}")

        # Inserir colunas genéricas para dados sem cabeçalhos
        try:
            data = [line.split() for line in data_lines]
            table = pd.DataFrame(data, columns=[f"Column_{i+1}" for i in range(max_columns)])
            return table
        except Exception as fallback_error:
            print(f"Erro no fallback: {fallback_error}")
            return None

# ------------------------------
# Execução Principal
# ------------------------------
# Padrão de cabeçalhos
header_pattern = r"^\d+\s+\*{5}\s*THE USAF AUTOMATED MISSILE DATCOM\s*\*\s*REV 3/99\s*\*{5}"

# Ler o arquivo
file_path = "Dat/for006.dat"
with open(file_path, 'r') as file:
    file_lines = file.readlines()

# Identificar blocos no arquivo
blocks = identify_blocks(file_lines, header_pattern)

# Resumo dos blocos detectados
block_summary = [(header, len(block)) for header, block in blocks]
block_summary_df = pd.DataFrame(block_summary, columns=["Header", "Number of Lines"])

# Exibir o resumo no console
print("\nResumo dos Blocos Detectados:")
print(block_summary_df)

# ------------------------------
# Extrair Tabelas e Salvar em CSV
# ------------------------------
extracted_tables = {}
output_dir = "Dat/out/extracted_tables/"
os.makedirs(output_dir, exist_ok=True)

for header, block in blocks:
    print(f"Processando bloco com cabeçalho: {header}")
    tables = extract_tables_general(block)
    extracted_tables[header] = tables

    # Salvar tabelas como CSV
    sanitized_header = re.sub(r'[^\w\s]', '', header).replace(" ", "_")[:50]  # Sanitizar nome do arquivo
    for i, table in enumerate(tables):
        if table is not None and not table.empty:
            output_path = os.path.join(output_dir, f"{sanitized_header}_table_{i+1}.csv")
            table.to_csv(output_path, index=False)

# ------------------------------
# Resumo das Tabelas Extraídas
# ------------------------------
tables_summary = []
for header, tables in extracted_tables.items():
    for i, table in enumerate(tables):
        tables_summary.append((header, i + 1, len(table), list(table.columns) if not table.empty else []))

tables_summary_df = pd.DataFrame(tables_summary, columns=["Header", "Table Number", "Number of Rows", "Columns"])

# Exibir o resumo das tabelas no console
print("\nResumo das Tabelas Extraídas:")
print(tables_summary_df)

# Diretório onde as tabelas foram salvas
print(f"\nTabelas salvas no diretório: {output_dir}")
