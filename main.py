import tkinter as tk
from interfaz import AnalizadorFunciones
from logica import analizar_funcion, calcular_intersecciones, describir_resultados
from evaluacion import evaluar_punto

class AnalizadorFuncionesIntegrado(AnalizadorFunciones):
    def analyze(self):
        func_str = self.entry_function.get().strip()
        x_str = self.entry_x.get().strip()

        if not func_str:
            tk.messagebox.showwarning("Aviso", "tiene que ingresar una funcion")
            return

        try:
            expr, x = analizar_funcion(func_str)
            resultados = describir_resultados(expr)
            inter = calcular_intersecciones(expr)
            if x_str:
                x_val = float(x_str)
                eval_result = evaluar_punto(expr, x_val)
                resultados += "\n--- Evaluación paso a paso ---\n"
                for paso in eval_result.pasos:
                    resultados += paso + "\n"
                if eval_result.fuera_de_dominio:
                    resultados += "\n→ El punto está fuera del dominio.\n"
                else:
                    resultados += f"\nPar ordenado: ({x_val}, {eval_result.decimal})\n"
                self.show_graph(expr, inter["x"], inter["y"], (x_val, eval_result.decimal))
            else:
                self.show_graph(expr, inter["x"], inter["y"])
            self.text_results.delete("1.0", tk.END)
            self.text_results.insert(tk.END, resultados)
        except Exception as e:
            tk.messagebox.showerror("Error", f"No se pudo analizar tu funcion\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalizadorFuncionesIntegrado(root)
    root.mainloop()