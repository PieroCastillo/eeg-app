#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <thread>
#include <string>
#include <stdexcept>
#include <fstream>
#include <format>
#include <mutex>          // Para compartir datos entre hilos
#include <deque>          // Búfer eficiente para los datos
#include <atomic>         // Para detener el hilo de forma segura
#include <numeric>        // Para std::iota
#include <algorithm>      // Para std::copy

// --- Dependencias de GUI ---
// Estas bibliotecas deben estar en tu sistema
// (ver README.md para instrucciones de instalación)
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#endif

// -----------------------------------------------
// Constantes y Estado Global
// -----------------------------------------------
const int NUM_CHANNELS = 3;
const int PLOT_BUFFER_SIZE = 1024; // Muestras a mostrar en el gráfico

// Búferes de datos compartidos (uno para cada canal)
std::vector<std::deque<float>> g_channelData(NUM_CHANNELS);
std::mutex g_dataMutex; // Protege el acceso a g_channelData
std::atomic<bool> g_keepRunning = true; // Señal para detener el hilo
std::atomic<double> g_sampleFreq = 0.0; // Frecuencia de muestreo

// NUEVO: Estado para el zoom (escala vertical) de los gráficos
struct PlotRange {
    float min;
    float max;
};
// Inicializa los rangos por defecto (AHORA COMPARTIDOS)
PlotRange g_sharedRawPlotRange = {0.0f, 1023.0f};
PlotRange g_sharedDwtPlotRange = {-512.0f, 512.0f};


// -----------------------------------------------
// Decodifica un frame binario (2 bytes por canal)
// (Tu función original, sin cambios)
// -----------------------------------------------
std::vector<int> decodeFrame(const std::vector<uint8_t>& frame, int numChannels)
{
    std::vector<int> values(numChannels);
    for (int i = 0; i < numChannels; ++i)
    {
        uint8_t b1 = frame[i * 2];
        uint8_t b2 = frame[i * 2 + 1];
        int val = ((b1 & 0x07) << 7) | (b2 & 0x7F);
        values[i] = val;
    }
    return values;
}

// -----------------------------------------------
// Clase SerialPort multiplataforma simple
// (Tu clase original, sin cambios)
// -----------------------------------------------
class SerialPort {
public:
    SerialPort(const std::string& portName, int baudRate)
    {
#ifdef _WIN32
        handle = CreateFileA(portName.c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
        if (handle == INVALID_HANDLE_VALUE)
            throw std::runtime_error("No se pudo abrir el puerto serial");

        DCB dcbSerialParams = { 0 };
        dcbSerialParams.DCBlength = sizeof(dcbSerialParams);
        if (!GetCommState(handle, &dcbSerialParams))
            throw std::runtime_error("Error obteniendo estado del puerto");

        dcbSerialParams.BaudRate = baudRate;
        dcbSerialParams.ByteSize = 8;
        dcbSerialParams.StopBits = ONESTOPBIT;
        dcbSerialParams.Parity = NOPARITY;
        SetCommState(handle, &dcbSerialParams);

        COMMTIMEOUTS timeouts = { 0 };
        // Tiempos de espera más cortos para una GUI responsiva
        timeouts.ReadIntervalTimeout = 1; 
        timeouts.ReadTotalTimeoutConstant = 1;
        timeouts.ReadTotalTimeoutMultiplier = 1;
        SetCommTimeouts(handle, &timeouts);
#else
        fd = open(portName.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
        if (fd < 0) throw std::runtime_error("No se pudo abrir el puerto serial");

        struct termios tty {};
        if (tcgetattr(fd, &tty) != 0)
            throw std::runtime_error("Error configurando termios");
        
        // El baudrate B230400 está en termios.h
        cfsetospeed(&tty, B230400);
        cfsetispeed(&tty, B230400);

        tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
        tty.c_iflag = 0;
        tty.c_oflag = 0;
        tty.c_lflag = 0;
        tty.c_cc[VMIN]  = 0; // Lectura no bloqueante
        tty.c_cc[VTIME] = 1; // 0.1 segundos de timeout

        tty.c_cflag |= (CLOCAL | CREAD);
        tty.c_cflag &= ~(PARENB | PARODD);
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CRTSCTS;

        if (tcsetattr(fd, TCSANOW, &tty) != 0)
            throw std::runtime_error("Error aplicando configuración del puerto");
#endif
    }

    ~SerialPort() {
#ifdef _WIN32
        if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
#else
        if (fd >= 0) close(fd);
#endif
    }

    int read(uint8_t* buffer, int size)
    {
#ifdef _WIN32
        DWORD bytesRead;
        if (!ReadFile(handle, buffer, size, &bytesRead, NULL))
            return -1;
        return static_cast<int>(bytesRead);
#else
        return ::read(fd, buffer, size);
#endif
    }

private:
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
};

// -----------------------------------------------
// Implementación de Wavelet (Haar DWT)
// -----------------------------------------------
/**
 * @brief Aplica una Transformada Rápida de Haar (DWT) in-place.
 * @param data Vector de datos. La longitud debe ser potencia de 2.
 * @param n Longitud de la sección a transformar (para recursión).
 */
void haart_inplace(std::vector<float>& data, int n) {
    if (n < 2) return;

    std::vector<float> temp(n);
    int half = n / 2;
    for (int i = 0; i < half; ++i) {
        // Coeficiente de aproximación (promedio)
        temp[i] = (data[2 * i] + data[2 * i + 1]) / 2.0f;
        // Coeficiente de detalle (diferencia)
        temp[i + half] = (data[2 * i] - data[2 * i + 1]) / 2.0f;
    }

    // Copia los resultados temporales de vuelta
    std::copy(temp.begin(), temp.end(), data.begin());

    // Recursión sobre la parte de aproximación
    haart_inplace(data, half);
}

/**
 * @brief Prepara los datos y llama a la transformada de Haar.
 * @param data_in Vector con datos de entrada.
 * @return Un nuevo vector con los coeficientes de la DWT.
 */
std::vector<float> compute_haart_dwt(const std::vector<float>& data_in) {
    if (data_in.empty()) return {};

    // 1. Encontrar la mayor potencia de 2 <= tamaño de los datos
    int n = 1;
    while (n * 2 <= data_in.size()) {
        n *= 2;
    }
    if (n < 2) return {}; // No hay suficientes datos para transformar

    // 2. Copiar la sección más reciente de datos (tamaño n)
    std::vector<float> data_out(n);
    std::copy(data_in.end() - n, data_in.end(), data_out.begin());
    
    // 3. Aplicar la transformada in-place
    haart_inplace(data_out, n);
    
    return data_out;
}


// -----------------------------------------------
// Función del Hilo Lector
// -----------------------------------------------
void readSpikeShield_thread(const std::string& port)
{
    try {
        SerialPort serial(port, 230400);
        std::vector<uint8_t> buffer;
        buffer.reserve(2048);

        std::cout << std::format("📡 [Hilo Lector] Leyendo {} canal(es) desde {}...\n", NUM_CHANNELS, port);

        using clock = std::chrono::high_resolution_clock;
        auto lastTime = clock::now();
        int samples = 0;

        while (g_keepRunning)
        {
            uint8_t temp[512];
            int n = serial.read(temp, sizeof(temp));
            if (n > 0) buffer.insert(buffer.end(), temp, temp + n);

            // Bucle para procesar todos los frames completos en el búfer
            while (buffer.size() >= 2 * NUM_CHANNELS)
            {
                // 1. Sincronizar: buscar el byte de cabecera (MSB = 1)
                if (buffer[0] & 0x80)
                {
                    // 2. Tenemos un frame. Decodificar.
                    std::vector<uint8_t> frame(buffer.begin(), buffer.begin() + 2 * NUM_CHANNELS);
                    buffer.erase(buffer.begin(), buffer.begin() + 2 * NUM_CHANNELS);

                    auto values = decodeFrame(frame, NUM_CHANNELS);
                    samples++;

                    // 3. Guardar los datos en el búfer global
                    {
                        std::lock_guard<std::mutex> lock(g_dataMutex);
                        for (int i = 0; i < NUM_CHANNELS; ++i) {
                            g_channelData[i].push_back(static_cast<float>(values[i]));
                            // Mantener el búfer al tamaño máximo
                            if (g_channelData[i].size() > PLOT_BUFFER_SIZE) {
                                g_channelData[i].pop_front();
                            }
                        }
                    }

                    // 4. Calcular frecuencia (solo en este hilo)
                    if (samples >= 100)
                    {
                        auto now = clock::now();
                        double dt = std::chrono::duration<double>(now - lastTime).count();
                        g_sampleFreq = samples / dt; // Actualiza el atomic
                        lastTime = now;
                        samples = 0;
                    }
                }
                else {
                    // Byte incorrecto, descartar y seguir buscando
                    buffer.erase(buffer.begin());
                }
            }
            
            // Dormir brevemente para ceder tiempo de CPU si no hay datos
            if (n <= 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "❌ Error en el hilo lector: " << e.what() << '\n';
        g_keepRunning = false; // Detiene la aplicación principal
    }
    std::cout << "🔌 [Hilo Lector] Detenido.\n";
}

// -----------------------------------------------
// Funciones de Ayuda de GLFW
// -----------------------------------------------
static void glfw_error_callback(int error, const char* description)
{
    std::cerr << std::format("Error de GLFW {}: {}\n", error, description);
}

// -----------------------------------------------
// main (Hilo Principal / GUI)
// -----------------------------------------------
int main()
{
    // --- 1. Inicializar GLFW y Ventana ---
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // GL 3.3 + GLSL 330
    const char* glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Visor SpikeShield (ImGui)", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // --- 2. Inicializar GLAD ---
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Error: No se pudo inicializar GLAD\n";
        return 1;
    }

    // --- 3. Inicializar ImGui ---
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.Fonts->AddFontDefault();
    io.FontGlobalScale = 1.25f; // Letras un poco más grandes
    
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // --- 4. Iniciar Hilo Lector ---
    std::string portName;
#ifdef _WIN32
    portName = "COM5";
#else
    portName = "/dev/ttyUSB0";
#endif
    std::thread serialThread(readSpikeShield_thread, portName);

    // --- 5. Bucle Principal de Renderizado ---
    while (!glfwWindowShouldClose(window) && g_keepRunning)
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // --- 6. Dibujar la GUI ---
        {
            // Hacemos una ventana de ImGui que ocupe toda la ventana de GLFW
            int width, height;
            glfwGetWindowSize(window, &width, &height);
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImVec2(width, height));
            ImGui::Begin("Visor de Canales", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

            ImGui::Text("Puerto: %s", portName.c_str());
            ImGui::SameLine();
            ImGui::Text(" | Frecuencia: %.1f Hz", g_sampleFreq.load());
            ImGui::Separator();

            // --- Controles de Zoom Globales ---
            ImGui::DragFloatRange2("Zoom Y (Raw Global)", &g_sharedRawPlotRange.min, &g_sharedRawPlotRange.max, 1.0f, -2048.0f, 2048.0f);
            ImGui::SameLine();
            ImGui::DragFloatRange2("Zoom Y (DWT Global)", &g_sharedDwtPlotRange.min, &g_sharedDwtPlotRange.max, 1.0f, -1024.0f, 1024.0f);
            ImGui::Separator(); // Otro separador

            // Copiamos los datos de los búferes globales para evitar
            // mantener el mutex bloqueado mientras dibujamos.
            std::vector<std::vector<float>> plotData(NUM_CHANNELS);
            {
                std::lock_guard<std::mutex> lock(g_dataMutex);
                for (int i = 0; i < NUM_CHANNELS; ++i) {
                    // Copia de deque a vector
                    plotData[i].assign(g_channelData[i].begin(), g_channelData[i].end());
                }
            }
            
            // Ancho del gráfico (casi toda la ventana)
            float plotWidth = ImGui::GetContentRegionAvail().x;
            // Ajustamos la altura: (2 Textos + Separador) ~ 30px por canal?
            // El espacio para los sliders globales ya se restó de GetContentRegionAvail().y
            float plotHeight = (ImGui::GetContentRegionAvail().y - (NUM_CHANNELS * 30)) / (NUM_CHANNELS * 2.0f); // Dividir espacio
            if (plotHeight < 50.0f) plotHeight = 50.0f; // Mínimo

            for (int i = 0; i < NUM_CHANNELS; ++i)
            {
                ImGui::PushID(i); // ID único para widgets de este canal

                if (plotData[i].empty()) {
                    ImGui::Text("Canal %d: Sin datos...", i + 1);
                    ImGui::PopID(); // No olvidar el PopID
                    continue;
                }

                // --- Gráfico de Señal Cruda ---
                ImGui::Text("Canal %d - Señal Cruda (%zu muestras)", i + 1, plotData[i].size());

                // El slider de zoom ahora es global (está fuera del bucle)

                ImGui::PlotLines("", plotData[i].data(), (int)plotData[i].size(), 0, 
                                 nullptr, 
                                 g_sharedRawPlotRange.min, g_sharedRawPlotRange.max, // Usar estado de zoom COMPARTIDO
                                 ImVec2(plotWidth, plotHeight));
                
                // --- Gráfico de Wavelet ---
                std::vector<float> waveletCoeffs = compute_haart_dwt(plotData[i]);
                if (!waveletCoeffs.empty()) {
                    ImGui::Text("Canal %d - Haar DWT (%zu coefs)", i + 1, waveletCoeffs.size());
                    
                    // El slider de zoom ahora es global (está fuera del bucle)

                    ImGui::PlotLines("", waveletCoeffs.data(), (int)waveletCoeffs.size(), 0,
                                     nullptr, 
                                     g_sharedDwtPlotRange.min, g_sharedDwtPlotRange.max, // Usar estado de zoom COMPARTIDO
                                     ImVec2(plotWidth, plotHeight));
                }
                
                ImGui::Separator();
                ImGui::PopID(); // Fin de ID único
            }

            ImGui::End();
        }

        // --- 7. Renderizado ---
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // --- 8. Limpieza ---
    std::cout << "Cerrando aplicación...\n";
    g_keepRunning = false;
    if(serialThread.joinable())
        serialThread.join();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}