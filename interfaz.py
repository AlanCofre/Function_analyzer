import tkinter as tk
from tkinter import messagebox
from sympy import symbols, sympify, lambdify, solveset, S
from PIL import Image, ImageTk
import numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

x = symbols('x')

class AnalizadorFunciones:

    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de funciones")
        self.root.geometry("720x500")
        self.create_widgets()

    def create_widgets(self):
        title_label = tk.Label(
            self.root,
            text="Analizador de funciones",
            pady=10
            )

        image = tk.Frame(self.root)
        image.pack(expand=False)
        try:
            img = Image.open("analizador.jpg")
            img = img.resize((100, 100))
            self.imagen = ImageTk.PhotoImage(img)
            tk.Label(image, image=self.imagen).pack(padx=10)
        except:
            tk.Label(image, text="[imagen]").pack(padx=10)
        title_label.pack()

        frame_func = tk.Frame(self.root, pady=10)
        frame_func.pack(padx=20)
        tk.Label(frame_func, text="Ingrese f(x):").pack(side="left")
        self.entry_function = tk.Entry(frame_func, width=30)
        self.entry_function.pack(padx=10)

        frame_x = tk.Frame(self.root, pady=10)
        frame_x.pack(padx=20)
        tk.Label(frame_x, text="Valor de x (opcional):").pack(side="left")
        self.entry_x = tk.Entry(frame_x, width=10)
        self.entry_x.pack(padx=10)

        #botones
        frame_btn = tk.Frame(self.root, pady=15)
        frame_btn.pack()
        btn_analyze = tk.Button(
            frame_btn, text="Analizar funcion",
            command=self.analyze, bg="#00F108", fg="white", width=15
        )
        btn_analyze.grid(row=0, column=0, padx=10)

        btn_clear = tk.Button(
            frame_btn, text="Limpiar",
            command=self.clear_inputs, bg="#ff1100", fg="white", width=10
        )
        btn_clear.grid(row=0, column=1, padx=10)

        results_frame = tk.LabelFrame(self.root, text="Resultados")
        results_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.text_results = tk.Text(results_frame, wrap="word", height=10)
        self.text_results.pack(fill="both", expand=True, padx=5, pady=5)

    def clear_inputs(self):
        self.entry_function.delete(0, tk.END)
        self.entry_x.delete(0, tk.END)
        self.text_results.delete("1.0", tk.END)

    def analyze(self):
        func_str = self.entry_function.get().strip()
        x_str = self.entry_x.get().strip()

        if not func_str:
            messagebox.showwarning("Aviso", "tiene que ingresar una funcion")
            return

        try:
            expr = sympify(func_str)
            intersecciones_x = solveset(expr, x, domain=S.Reals)
            interseccion_y = expr.subs(x, 0)

            result_text = f"funcion ingresada: f(x) = {expr}\n"
            result_text += f"intersecciones con eje X: {list(intersecciones_x)}\n"
            result_text += f"intersección con eje Y: {interseccion_y}\n"

            if x_str:
                x_val = float(x_str)
                y_val = expr.subs(x, x_val)
                result_text += f"evaluacion: f({x_val}) = {y_val}\n"
                self.show_graph(expr, intersecciones_x, interseccion_y, (x_val, y_val))
            else:
                self.show_graph(expr, intersecciones_x, interseccion_y)

            self.text_results.delete("1.0", tk.END)
            self.text_results.insert(tk.END, result_text)

        except Exception as e:
            messagebox.showerror("Error", f"no se pudo analizar tu funcion\n{e}")
    #grafico
    def show_graph(self, expr, inters_x, inters_y, punto_eval=None):
        graph_win = tk.Toplevel(self.root)
        graph_win.title("Grafico")
        graph_win.geometry("840x620")

#        f = lambdify(x, expr, 'numpy')
#        X = numpy.linspace(-10, 10, 400)
#        Y = f(X)
#
#        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
#        ax.plot(X, Y, label=f"f(x) = {expr}", color="blue")
#        ax.axhline(0, color="black", linewidth=1)
#        ax.axvline(0, color="black", linewidth=1)
#
#       for val in inters_x:
#            try:
#                val_num = float(val)
#                ax.scatter(val_num, 0, color="green", s=60, label="Interseccion X")
#            except:
#                pass
#
#
#        try:
#            ax.scatter(0, float(inters_y), color="blue", s=60, label="Interseccion Y")
#        except:
#            pass
#
#        if punto_eval:
#            ax.scatter(punto_eval[0], punto_eval[1], color="red", s=80, label=f"Punto evaluado ({punto_eval[0]}, {punto_eval[1]})")
#
#       ax.set_title("Grafica de f(x)", fontsize=14)
#        ax.set_xlabel("Eje X")
#        ax.set_ylabel("Eje Y")
#        ax.legend()
#        ax.grid(True)
#
#        canvas = FigureCanvasTkAgg(fig, master=graph_win)
#        canvas.draw()
#        canvas.get_tk_widget().pack(fill="both", expand=True)
if __name__ == "__main__":
    root = tk.Tk()
    app = AnalizadorFunciones(root)
    root.mainloop()
