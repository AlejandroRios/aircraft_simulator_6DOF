import re
import pandas as pd

class DATCOMOutputReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.blocks = {}
    
    def read_file(self):
        """
        Lê o arquivo e organiza por blocos relevantes.
        """
        with open(self.filepath, "r") as f:
            lines = f.readlines()
        
        self.blocks = self._split_blocks(lines)
    
    def _split_blocks(self, lines):
        """
        Divide o arquivo em blocos baseados nos cabeçalhos principais.
        """
        blocks = {}
        current_block = []
        current_header = None
        
        for line in lines:
            if "STATIC AERODYNAMICS" in line or "FIN SET" in line:
                # Salvar o bloco anterior
                if current_block and current_header:
                    blocks[current_header] = current_block
                # Iniciar um novo bloco
                current_header = line.strip()
                current_block = []
            elif current_header:
                current_block.append(line)
        
        # Salvar o último bloco
        if current_block and current_header:
            blocks[current_header] = current_block
        
        return blocks
    
    def extract_flight_conditions(self, block):
        """
        Extrai condições de voo do bloco.
        """
        conditions = {}
        for line in block:
            if "MACH NO" in line:
                match = re.search(r"MACH NO\s*=\s*([\d.]+)", line)
                if match:
                    conditions["MACH"] = float(match.group(1))
            elif "ALTITUDE" in line:
                match = re.search(r"ALTITUDE\s*=\s*([\d.]+)", line)
                if match:
                    conditions["ALTITUDE"] = float(match.group(1))
        return conditions
    
    def extract_coefficients(self, block):
        """
        Extrai coeficientes estáticos (CN, CM, CA, etc.) do bloco.
        """
        data = []
        headers = []
        inside_table = False
        
        for line in block:
            if "ALPHA" in line and "CN" in line:
                headers = re.split(r"\s+", line.strip())
                inside_table = True
            elif inside_table and re.match(r"^\s*\d", line):
                values = re.split(r"\s+", line.strip())
                data.append(values)
            elif inside_table and not line.strip():
                break
        
        return pd.DataFrame(data, columns=headers)
    
    def extract_derivatives(self, block):
        """
        Extrai derivadas (CNA, CMA, etc.) do bloco.
        """
        data = []
        headers = []
        inside_table = False
        
        for line in block:
            if "ALPHA" in line and "CNA" in line:
                headers = re.split(r"\s+", line.strip())
                inside_table = True
            elif inside_table and re.match(r"^\s*\d", line):
                values = re.split(r"\s+", line.strip())
                data.append(values)
            elif inside_table and not line.strip():
                break
        
        return pd.DataFrame(data, columns=headers)
    
    def process_output(self):
        """
        Processa todas as seções relevantes do arquivo.
        """
        results = []
        
        for header, block in self.blocks.items():
            if "STATIC AERODYNAMICS" in header:
                flight_conditions = self.extract_flight_conditions(block)
                coefficients = self.extract_coefficients(block)
                results.append((flight_conditions, coefficients))
            elif "DERIVATIVES" in header:
                derivatives = self.extract_derivatives(block)
                results.append(derivatives)
        
        return results


# Exemplo de uso
reader = DATCOMOutputReader("output_exemplo.dat")
reader.read_file()
results = reader.process_output()

for item in results:
    print(item)