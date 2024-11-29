class MissileDATCOMInput:
    def __init__(self, nalpha=8, mach_vals=None, alt_vals=None, alpha_vals=None):
        """
        Classe para criar inputs do Missile DATCOM.

        Parâmetros:
        - nalpha: Número de ângulos de ataque (int).
        - mach_vals: Lista de números de Mach (list of floats). Se None, usa valores padrão.
        - alt_vals: Lista de altitudes (list of floats). Deve ter o mesmo tamanho que mach_vals.
        - alpha_vals: Lista de ângulos de ataque (list of floats). Se None, usa valores padrão.
        """
        self.nalpha = nalpha
        self.mach_vals = mach_vals if mach_vals else [2.36]  # Valor padrão para Mach
        self.alt_vals = alt_vals if alt_vals else [0.0]  # Altitude padrão (nível do mar)
        self.alpha_vals = alpha_vals if alpha_vals else [0., 4., 8., 12., 16., 20., 24., 28.]  # Valores padrão

        # Validação: Mach e Alt devem ter o mesmo tamanho
        if len(self.mach_vals) != len(self.alt_vals):
            raise ValueError("O número de valores de Mach deve ser igual ao número de valores de Altitude.")

    def gerar_input(self, filename="for005.dat"):
        """
        Gera o arquivo de input for005.dat.

        Parâmetros:
        - filename: Nome do arquivo a ser gerado (string).
        """
        with open(filename, "w") as f:
            # Cabeçalho
            f.write("***** THE USAF AUTOMATED MISSILE DATCOM *****\n")
            f.write("***** REV 03/11 *****\n\n")

            # Configuração inicial
            f.write("CASEID PLANAR WING, CRUCIFORM PLUS TAIL CONFIGURATION\n")
            f.write("$FLTCON\n")
            f.write(f" NALPHA={self.nalpha}.\n")
            f.write(f" NMACH={len(self.mach_vals)}.\n")
            f.write(f" MACH={','.join(map(str, self.mach_vals))}\n")
            f.write(f" ALT={','.join(map(str, self.alt_vals))}\n")
            f.write(f" ALPHA={','.join(map(str, self.alpha_vals))}\n")
            f.write("$\n")

            # Configurações adicionais fixas
            f.write("$REFQ\n")
            f.write(" XCG=18.75\n")
            f.write("$\n")

            f.write("$AXIBOD\n")
            f.write(" LNOSE=11.25\n")
            f.write(" DNOSE=3.75\n")
            f.write(" LCENTR=26.25\n")
            f.write(" DEXIT=0.\n")
            f.write("$\n")

            f.write("$FINSET1\n")
            f.write(" XLE=15.42\n")
            f.write(" NPANEL=2.\n")
            f.write(" PHIF=90.,270.\n")
            f.write(" SWEEP=0.\n")
            f.write(" STA=1.\n")
            f.write(" CHORD=6.\n")
            f.write(" SSPAN=1.875,5.355\n")
            f.write(" ZUPPER=0.02238,0.02238\n")
            f.write(" LAMAXU=0.238,0.238\n")
            f.write(" LFLATU=0.524,0.524\n")
            f.write(" LER=0.015,0.015\n")
            f.write("$\n")

            f.write("$FINSET2\n")
            f.write(" XLE=31.915\n")
            f.write(" NPANEL=4.\n")
            f.write(" PHIF=0.,90.,180.,270.\n")
            f.write(" SWEEP=0.\n")
            f.write(" STA=1.\n")
            f.write(" SSPAN=1.875,6.26\n")
            f.write(" CHORD=5.585,2.792\n")
            f.write(" ZUPPER=0.02238,0.02238\n")
            f.write(" LAMAXU=0.288,0.288\n")
            f.write(" LFLATU=0.428,0.428\n")
            f.write("$\n")

            f.write("PART\n")
            f.write("PLOT\n")
            f.write("DAMP\n")
            f.write("SAVE\n")
            f.write("DIM IN\n")
            f.write("NEXT CASE\n")
            f.write("CASEID TRIM OF CASE NUMBER 1\n")
            f.write("$TRIM\n")
            f.write(" SET=2.\n")
            f.write("$\n")
            f.write("PRINT AERO TRIM\n")
            f.write("NEXT CASE\n")

        print(f"Arquivo {filename} gerado com sucesso!")

# Exemplo de uso
# Configuração padrão
input_padrao = MissileDATCOMInput()
input_padrao.gerar_input()

# Configuração customizada
input_customizado = MissileDATCOMInput(
    nalpha=6,
    mach_vals=[0.8, 1.5, 2.0],
    alt_vals=[1000, 5000, 10000],
    alpha_vals=[0., 2., 4., 6., 8., 10.]
)
input_customizado.gerar_input("for005_custom.dat")
