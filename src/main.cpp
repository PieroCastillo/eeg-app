#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <thread>
#include <string>
#include <stdexcept>
#include <format>
#include <mutex>          // Para compartir datos entre hilos
#include <deque>          // Búfer eficiente para los datos
#include <atomic>         // Para detener el hilo de forma segura
#include <numeric>        // Para std::iota, std::accumulate
#include <algorithm>      // Para std::copy, std::max_element
#include <cmath>          // Para std::sqrt
#include <limits>         // Para std::numeric_limits

// --- Dependencias de GUI ---
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#ifdef _WIN32
#define NOMINMAX
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
const int PLOT_BUFFER_SIZE = 1024; // Muestras a usar para DWT (debe ser potencia de 2)
const int ENERGY_HISTORY_SIZE = 300; // Número de frames de energía para el espectrograma (Avanza más lento)
const int DWT_COEFFICIENTS_TO_SHOW = 512; // Mostrar hasta Gamma baja

// Estructura para definir las bandas cerebrales (EEG)
struct BrainwaveBand {
    std::string name;
    float freqMin; // Frecuencia Min (Hz)
    float freqMax; // Frecuencia Max (Hz)
    ImVec4 color;  // Color base para el fondo/líneas
    int startIdx;  // Índice de inicio en el vector DWT (para N=1024)
    int endIdx;    // Índice de fin en el vector DWT (para N=1024)
};

// Definición de bandas (Mapeo heurístico para DWT de 1024 coeficientes)
const std::vector<BrainwaveBand> g_bands = {
    // Indices: [0-31] -> A5/Delta
    {"Delta (<4 Hz) [A5]", 0.0f, 4.0f, ImVec4(0.0f, 0.5f, 1.0f, 1.0f), 0, 31},    // Azul
    // Indices: [32-63] -> D5/Theta
    {"Theta (4-8 Hz) [D5]", 4.0f, 8.0f, ImVec4(0.0f, 1.0f, 0.0f, 1.0f), 32, 63},   // Verde
    // Indices: [64-127] -> D4/Alpha
    {"Alpha (8-13 Hz) [D4]", 8.0f, 13.0f, ImVec4(1.0f, 1.0f, 0.0f, 1.0f), 64, 127}, // Amarillo
    // Indices: [128-255] -> D3/Beta
    {"Beta (13-30 Hz) [D3]", 13.0f, 30.0f, ImVec4(1.0f, 0.5f, 0.0f, 1.0f), 128, 255}, // Naranja
    // Indices: [256-511] -> D2/Gamma Baja
    {"Gamma (30-60 Hz) [D2]", 30.0f, 60.0f, ImVec4(1.0f, 0.0f, 0.5f, 1.0f), 256, 511}  // Rojo/Magenta
};


// Búferes de datos compartidos
std::vector<std::deque<float>> g_channelData(NUM_CHANNELS);
// Historial de energías de banda para el espectrograma
std::vector<std::deque<std::vector<float>>> g_bandEnergyHistory(NUM_CHANNELS);
std::mutex g_dataMutex; // Protege el acceso a los búferes
std::atomic<bool> g_keepRunning = true; // Señal para detener el hilo
std::atomic<double> g_sampleFreq = 0.0; // Frecuencia de muestreo

// Zoom global para los gráficos DWT
struct PlotRange { float min; float max; };
PlotRange g_sharedDwtPlotRange = {-512.0f, 512.0f};


// -----------------------------------------------
// Decodifica un frame binario (sin cambios)
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
// Clase SerialPort (sin cambios)
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
        
        cfsetospeed(&tty, B230400);
        cfsetispeed(&tty, B230400);

        tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
        tty.c_iflag = 0;
        tty.c_oflag = 0;
        tty.c_lflag = 0;
        tty.c_cc[VMIN]  = 0;
        tty.c_cc[VTIME] = 1;

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
// Implementación de Wavelet (Haar DWT) - (sin cambios)
// -----------------------------------------------
void haart_inplace(std::vector<float>& data, int n) {
    if (n < 2) return;

    std::vector<float> temp(n);
    int half = n / 2;
    for (int i = 0; i < half; ++i) {
        temp[i] = (data[2 * i] + data[2 * i + 1]) / 2.0f;
        temp[i + half] = (data[2 * i] - data[2 * i + 1]) / 2.0f;
    }

    std::copy(temp.begin(), temp.end(), data.begin());
    haart_inplace(data, half);
}

std::vector<float> compute_haart_dwt(const std::vector<float>& data_in) {
    if (data_in.empty()) return {};

    int n = 1;
    while (n * 2 <= data_in.size()) {
        n *= 2;
    }
    if (n < PLOT_BUFFER_SIZE) return {}; 

    std::vector<float> data_out(PLOT_BUFFER_SIZE);
    std::copy(data_in.end() - PLOT_BUFFER_SIZE, data_in.end(), data_out.begin());
    
    haart_inplace(data_out, PLOT_BUFFER_SIZE);
    
    return data_out;
}

std::vector<float> calculate_band_energy(const std::vector<float>& dwtCoeffs) {
    std::vector<float> bandEnergies;
    if (dwtCoeffs.empty()) return bandEnergies;

    for (const auto& band : g_bands) {
        float sumOfSquares = 0.0f;
        int count = 0;
        
        int actualEnd = std::min(band.endIdx, (int)dwtCoeffs.size() - 1);

        for (int i = band.startIdx; i <= actualEnd; ++i) {
            sumOfSquares += dwtCoeffs[i] * dwtCoeffs[i];
            count++;
        }

        float energy = 0.0f;
        if (count > 0) {
            energy = std::sqrt(sumOfSquares / count);
        }
        bandEnergies.push_back(energy);
    }
    return bandEnergies;
}

// -----------------------------------------------
// Función del Hilo Lector (sin cambios funcionales)
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

            while (buffer.size() >= 2 * NUM_CHANNELS)
            {
                if (buffer[0] & 0x80)
                {
                    std::vector<uint8_t> frame(buffer.begin(), buffer.begin() + 2 * NUM_CHANNELS);
                    buffer.erase(buffer.begin(), buffer.begin() + 2 * NUM_CHANNELS);

                    auto values = decodeFrame(frame, NUM_CHANNELS);
                    samples++;

                    {
                        std::lock_guard<std::mutex> lock(g_dataMutex);
                        for (int i = 0; i < NUM_CHANNELS; ++i) {
                            g_channelData[i].push_back(static_cast<float>(values[i]));
                            if (g_channelData[i].size() > PLOT_BUFFER_SIZE) {
                                g_channelData[i].pop_front();
                            }

                            if (g_channelData[i].size() == PLOT_BUFFER_SIZE) {
                                std::vector<float> recentData(g_channelData[i].begin(), g_channelData[i].end());
                                
                                std::vector<float> dwtCoeffs = compute_haart_dwt(recentData);
                                std::vector<float> bandEnergies = calculate_band_energy(dwtCoeffs);

                                if (!bandEnergies.empty()) {
                                    g_bandEnergyHistory[i].push_back(bandEnergies);
                                    if (g_bandEnergyHistory[i].size() > ENERGY_HISTORY_SIZE) {
                                        g_bandEnergyHistory[i].pop_front();
                                    }
                                }
                            }
                        }
                    }

                    if (samples >= 100)
                    {
                        auto now = clock::now();
                        double dt = std::chrono::duration<double>(now - lastTime).count();
                        g_sampleFreq = samples / dt; 
                        lastTime = now;
                        samples = 0;
                    }
                }
                else {
                    buffer.erase(buffer.begin());
                }
            }
            
            if (n <= 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "❌ Error en el hilo lector: " << e.what() << '\n';
        g_keepRunning = false;
    }
    std::cout << "🔌 [Hilo Lector] Detenido.\n";
}

// -----------------------------------------------
// Funciones de Ayuda de GLFW (sin cambios)
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
    // ... (Inicialización GLFW, GLAD, ImGui) ...
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    const char* glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Visor de Canales DWT y Espectrograma", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Error: No se pudo inicializar GLAD\n";
        return 1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.Fonts->AddFontDefault();
    io.FontGlobalScale = 1.25f;
    
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    // ------------------------------------------

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
        // Añadir una pequeña pausa para estabilidad durante el resize
        if (glfwGetWindowAttrib(window, GLFW_FOCUSED) == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // --- 6. Dibujar la GUI ---
        {
            int width, height;
            glfwGetWindowSize(window, &width, &height);
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImVec2(width, height));
            
            // Usar ImGuiWindowFlags_NoScrollbar para forzar que el contenido quepa en el cálculo de altura
            ImGui::Begin("Visor de Canales DWT y Espectrograma", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar);

            ImGui::Text("Puerto: %s", portName.c_str());
            ImGui::SameLine();
            ImGui::Text(" | Frecuencia de Muestreo (Fs): %.1f Hz", g_sampleFreq.load());
            ImGui::Separator();

            // --- Controles de Zoom Globales (Solo DWT) ---
            ImGui::DragFloatRange2("Zoom Y (DWT Global)", &g_sharedDwtPlotRange.min, &g_sharedDwtPlotRange.max, 1.0f, -4096.0f, 4096.0f);
            ImGui::Separator();

            // Copiamos los datos de los búferes globales
            std::vector<std::vector<float>> plotData(NUM_CHANNELS);
            std::vector<std::vector<std::vector<float>>> energyHistory(NUM_CHANNELS);
            {
                std::lock_guard<std::mutex> lock(g_dataMutex);
                for (int i = 0; i < NUM_CHANNELS; ++i) {
                    plotData[i].assign(g_channelData[i].begin(), g_channelData[i].end());
                    energyHistory[i].assign(g_bandEnergyHistory[i].begin(), g_bandEnergyHistory[i].end());
                }
            }
            
            float contentWidth = ImGui::GetContentRegionAvail().x;
            float reservedControlHeight = ImGui::GetCursorPosY() + 20; // Altura ocupada por texto y sliders superiores
            
            // Altura fija para cada gráfico DWT (más estable)
            const float DWT_PLOT_HEIGHT = 120.0f; 
            const float SPACER_HEIGHT = 5.0f;
            
            // Altura restante para el espectrograma
            float availableHeight = ImGui::GetContentRegionAvail().y;
            float requiredDwtHeight = (DWT_PLOT_HEIGHT + SPACER_HEIGHT) * NUM_CHANNELS;

            // El espectrograma solo se dibuja para el canal 1 (i=0)
            float spectrogramHeight = availableHeight - requiredDwtHeight - 5.0f; // 5.0f de margen
            if (spectrogramHeight < 100.0f) spectrogramHeight = 100.0f;

            for (int i = 0; i < NUM_CHANNELS; ++i)
            {
                ImGui::PushID(i); 

                if (plotData[i].size() < PLOT_BUFFER_SIZE) {
                    ImGui::Text("Canal %d: Esperando %d muestras para DWT...", i + 1, PLOT_BUFFER_SIZE);
                    ImGui::Separator();
                    ImGui::PopID();
                    continue;
                }

                // --- 1. Gráfico de Wavelet (con color por banda y acento de fondo) ---
                ImGui::Text("Canal %d - Coeficientes Haar DWT (N=%d)", i + 1, PLOT_BUFFER_SIZE);

                std::vector<float> waveletCoeffs = compute_haart_dwt(plotData[i]);
                std::vector<float> bandEnergies = calculate_band_energy(waveletCoeffs);
                
                if (!waveletCoeffs.empty()) {
                    
                    // a) Determinar banda predominante para acento de fondo
                    ImVec4 bgColor = ImGui::GetStyleColorVec4(ImGuiCol_FrameBg);
                    int predominantBandIndex = -1;
                    if (!bandEnergies.empty()) {
                        auto maxIt = std::max_element(bandEnergies.begin(), bandEnergies.end());
                        predominantBandIndex = std::distance(bandEnergies.begin(), maxIt);
                        
                        // Acentuar el color de la banda predominante
                        ImVec4 predominantColor = g_bands[predominantBandIndex].color;
                        bgColor = ImVec4(
                            bgColor.x + predominantColor.x * 0.15f,
                            bgColor.y + predominantColor.y * 0.15f,
                            bgColor.z + predominantColor.z * 0.15f,
                            1.0f
                        );
                        bgColor.x = std::min(bgColor.x, 1.0f);
                        bgColor.y = std::min(bgColor.y, 1.0f);
                        bgColor.z = std::min(bgColor.z, 1.0f);
                    }

                    // Usamos ImGui::BeginChild para un área de dibujo estable
                    ImGui::BeginChild("DWT_Plot_Child", ImVec2(contentWidth, DWT_PLOT_HEIGHT), true, ImGuiWindowFlags_NoScrollbar);
                    
                    // Dibujar fondo con acento de color
                    ImVec2 plot_min = ImGui::GetCursorScreenPos();
                    ImVec2 plot_max = ImVec2(plot_min.x + ImGui::GetContentRegionAvail().x, plot_min.y + ImGui::GetContentRegionAvail().y);
                    ImGui::GetWindowDrawList()->AddRectFilled(plot_min, plot_max, ImGui::GetColorU32(bgColor));

                    ImGui::Text("Banda Predominante: %s (RMS: %.2f)", 
                        (predominantBandIndex != -1) ? g_bands[predominantBandIndex].name.c_str() : "N/A",
                        (predominantBandIndex != -1) ? bandEnergies[predominantBandIndex] : 0.0f
                    );

                    // Plot de los coeficientes 
                    // Nota: PlotLines solo puede dibujar una línea. Se requiere dibujar múltiples líneas
                    // o usar AddPolyline para un color por punto/segmento preciso, lo cual es complejo.
                    // Mantendremos PlotLines sobre el array, y la coloración se indica por el fondo.
                    ImGui::PlotLines("", waveletCoeffs.data(), DWT_COEFFICIENTS_TO_SHOW, 0, 
                                     nullptr, 
                                     g_sharedDwtPlotRange.min, g_sharedDwtPlotRange.max,
                                     ImVec2(ImGui::GetContentRegionAvail().x, DWT_PLOT_HEIGHT - ImGui::GetTextLineHeightWithSpacing() * 2), sizeof(float));
                    
                    ImGui::EndChild();
                }
                
                ImGui::Dummy(ImVec2(contentWidth, SPACER_HEIGHT)); // Separador

                // --- 2. Espectrograma de Energía de Banda (Solo Canal 1 / i=0) ---
                if (i == 0) {
                    ImGui::Separator();
                    ImGui::Text("Canal %d: Espectrograma de Energía de Banda (Tiempo vs Frecuencia) - %d Frames", i + 1, ENERGY_HISTORY_SIZE);
                    
                    if (energyHistory[i].size() > 1) {
                        
                        ImGui::BeginChild("Spectrogram_Child", ImVec2(contentWidth, spectrogramHeight), true, ImGuiWindowFlags_NoScrollbar);
                        
                        float specWidth = ImGui::GetContentRegionAvail().x;
                        float specHeight = ImGui::GetContentRegionAvail().y;

                        float cellWidth = specWidth / ENERGY_HISTORY_SIZE;
                        float cellHeight = specHeight / g_bands.size();
                        
                        ImVec2 spec_min = ImGui::GetCursorScreenPos();
                        ImVec2 spec_max = ImVec2(spec_min.x + specWidth, spec_min.y + specHeight);
                        
                        // Normalización de la energía
                        float maxTotalEnergy = 0.0f;
                        for(const auto& frameEnergy : energyHistory[i]) {
                             // Usamos max para encontrar el valor máximo en todo el historial
                             for (float energy : frameEnergy) {
                                maxTotalEnergy = std::max(maxTotalEnergy, energy);
                             }
                        }
                        maxTotalEnergy = std::max(maxTotalEnergy, 1.0f);

                        // Dibujar el lienzo
                        ImGui::GetWindowDrawList()->AddRectFilled(spec_min, spec_max, ImGui::GetColorU32(ImVec4(0.1f, 0.1f, 0.1f, 1.0f)));
                        
                        // Dibujar Bandas y Espectrograma
                        for (size_t k = 0; k < g_bands.size(); ++k) {
                            const auto& band = g_bands[g_bands.size() - 1 - k]; // Iterar de Gamma (arriba) a Delta (abajo)
                            
                            // Etiqueta de Frecuencia/Banda
                            ImVec2 label_pos = ImVec2(spec_min.x + 5, spec_min.y + (k * cellHeight) + (cellHeight/3));
                            ImGui::GetWindowDrawList()->AddText(label_pos, ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)), band.name.c_str());

                            // Línea separadora de la banda
                            ImVec2 line_start = ImVec2(spec_min.x, spec_min.y + k * cellHeight);
                            ImVec2 line_end = ImVec2(spec_max.x, spec_min.y + k * cellHeight);
                            ImGui::GetWindowDrawList()->AddLine(line_start, line_end, ImGui::GetColorU32(ImVec4(0.3f, 0.3f, 0.3f, 1.0f)));

                            // Iterar sobre el tiempo (X-axis)
                            for (size_t t = 0; t < energyHistory[i].size(); ++t) {
                                const auto& frameEnergy = energyHistory[i][t];
                                size_t bandIdx = g_bands.size() - 1 - k; 
                                
                                if (bandIdx < frameEnergy.size()) {
                                    float energyNormalized = frameEnergy[bandIdx] / maxTotalEnergy;
                                    energyNormalized = std::min(energyNormalized, 1.0f);

                                    ImVec4 heatColor = band.color;
                                    ImVec4 finalColor = ImVec4(
                                        heatColor.x * energyNormalized * 0.8f,
                                        heatColor.y * energyNormalized * 0.8f,
                                        heatColor.z * energyNormalized * 0.8f,
                                        1.0f
                                    );
                                    
                                    ImVec2 cellA = ImVec2(spec_min.x + t * cellWidth, spec_min.y + k * cellHeight);
                                    ImVec2 cellB = ImVec2(spec_min.x + (t + 1) * cellWidth, spec_min.y + (k + 1) * cellHeight);

                                    ImGui::GetWindowDrawList()->AddRectFilled(cellA, cellB, ImGui::GetColorU32(finalColor));
                                }
                            }
                        }

                        // Etiqueta de Tiempo (X-axis)
                        ImGui::SetCursorScreenPos(ImVec2(spec_min.x + specWidth - 100, spec_max.y + 5));
                        ImGui::Text("Tiempo reciente ->");

                        ImGui::EndChild();
                    } else {
                        ImGui::Text("Calculando energía de bandas...");
                        ImGui::Dummy(ImVec2(contentWidth, spectrogramHeight));
                    }
                }
                
                ImGui::Separator();
                ImGui::PopID(); 
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