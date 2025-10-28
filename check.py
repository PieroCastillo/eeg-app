import serial
import time
from collections import deque

class SignalAnalyzer:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.channels = [deque(maxlen=window_size) for _ in range(3)]
        
    def add_sample(self, values):
        for i, value in enumerate(values):
            try:
                self.channels[i].append(float(value))
            except ValueError:
                print(f"\n[!] Error al convertir valor: {value}")
                
    def get_amplitudes(self):
        amplitudes = []
        for channel in self.channels:
            if len(channel) > 1:
                amplitude = max(channel) - min(channel)
                amplitudes.append(amplitude)
            else:
                amplitudes.append(0)
        return amplitudes

def monitor_serial():
    try:
        ser = serial.Serial(
            port='COM5',
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.2  # Lectura más reactiva
        )
        
        analyzer = SignalAnalyzer()
        print("✅ Conectado a COM5. Mostrando amplitudes en tiempo real...")
        print("Presione Ctrl+C para detener.\n")
        
        last_time = time.time()
        updates = 0

        while True:
            try:
                raw = ser.readline()
                if not raw:
                    continue  # Nada recibido
                
                # Intentamos decodificar de forma segura
                try:
                    decoded_line = raw.decode('utf-8', errors='ignore').strip()
                except UnicodeDecodeError:
                    decoded_line = raw.decode('latin1', errors='ignore').strip()
                
                if not decoded_line:
                    continue  # línea vacía
                
                # Filtrar caracteres no numéricos o de control
                if any(c in decoded_line for c in ['?', '#', '*', '$']):
                    continue

                # Intentar parsear
                values = decoded_line.replace(';', ',').split(',')
                values = [v.strip() for v in values if v.strip() != '']

                if len(values) == 3 and all(v.replace('.', '', 1).replace('-', '', 1).isdigit() for v in values):
                    analyzer.add_sample(values)
                    amplitudes = analyzer.get_amplitudes()
                    
                    updates += 1
                    current_time = time.time()
                    if current_time - last_time >= 1.0:
                        fps = updates / (current_time - last_time)
                        updates = 0
                        last_time = current_time

                        print(f"\rCh1: {amplitudes[0]:7.2f} | Ch2: {amplitudes[1]:7.2f} | Ch3: {amplitudes[2]:7.2f} | {fps:5.1f} Hz", end='')
                else:
                    # Mostrar solo si la línea no es numérica
                    if not decoded_line.replace('.', '', 1).isdigit():
                        print(f"\r[!] Datos inválidos: {decoded_line[:50]}", end='')
            
            except serial.SerialException as e:
                print(f"\n[!] Error de comunicación: {e}")
                break

            except Exception as e:
                print(f"\n[!] Error procesando datos: {e}")
                time.sleep(0.1)

            time.sleep(0.005)
            
    except serial.SerialException as e:
        print(f"\n❌ Error al abrir el puerto COM5: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Monitoreo detenido por el usuario.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("🔒 Puerto serial cerrado.")

if __name__ == "__main__":
    monitor_serial()
  