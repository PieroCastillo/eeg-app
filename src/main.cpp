#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <thread>
#include <string>
#include <stdexcept>
#include <format>
#include <mutex>
#include <deque>
#include <atomic>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <limits>

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
const int PLOT_BUFFER_SIZE = 1024;
const int ENERGY_HISTORY_SIZE = 300;
const int DWT_COEFFICIENTS_TO_SHOW = 512;

// Control de velocidad del espectrograma
const int SPECTROGRAM_UPDATE_INTERVAL = 10;
std::atomic<int> g_spectrogramUpdateCounter(0);

// NUEVO: Resolución del espectrograma (más bandas = más detalle)
const int SPECTROGRAM_FREQ_BANDS = 64; // Resolución vertical (64 bandas de frecuencia)

// Estructura para definir las bandas cerebrales (EEG)
struct BrainwaveBand {
    std::string name;
    float freqMin;
    float freqMax;
    ImVec4 color;
    int startIdx;
    int endIdx;
};

const std::vector<BrainwaveBand> g_bands = {
    {"Delta (<4 Hz) [A5]", 0.0f, 4.0f, ImVec4(0.0f, 0.5f, 1.0f, 1.0f), 0, 31},
    {"Theta (4-8 Hz) [D5]", 4.0f, 8.0f, ImVec4(0.0f, 1.0f, 0.0f, 1.0f), 32, 63},
    {"Alpha (8-13 Hz) [D4]", 8.0f, 13.0f, ImVec4(1.0f, 1.0f, 0.0f, 1.0f), 64, 127},
    {"Beta (13-30 Hz) [D3]", 13.0f, 30.0f, ImVec4(1.0f, 0.5f, 0.0f, 1.0f), 128, 255},
    {"Gamma (30-60 Hz) [D2]", 30.0f, 60.0f, ImVec4(1.0f, 0.0f, 0.5f, 1.0f), 256, 511}
};

// Búferes de datos compartidos
std::vector<std::deque<float>> g_channelData(NUM_CHANNELS);
std::vector<std::deque<std::vector<float>>> g_bandEnergyHistory(NUM_CHANNELS);
std::mutex g_dataMutex;
std::atomic<bool> g_keepRunning = true;
std::atomic<double> g_sampleFreq = 0.0;

// Cache de último DWT calculado
std::vector<std::vector<float>> g_lastDwtCoeffs(NUM_CHANNELS);
std::vector<std::vector<float>> g_lastBandEnergies(NUM_CHANNELS);
std::vector<bool> g_dwtNeedsUpdate(NUM_CHANNELS, true);

struct PlotRange { float min; float max; };
PlotRange g_sharedDwtPlotRange = {-512.0f, 512.0f};

// -----------------------------------------------
// NUEVO: Funciones de Suavizado e Interpolación
// -----------------------------------------------

// Interpolación bilineal para suavizar el espectrograma
float bilinearInterpolate(float q11, float q12, float q21, float q22, float tx, float ty) {
    float r1 = q11 * (1.0f - tx) + q21 * tx;
    float r2 = q12 * (1.0f - tx) + q22 * tx;
    return r1 * (1.0f - ty) + r2 * ty;
}

// Función de suavizado gaussiano simple para el historial
std::vector<std::vector<float>> smoothSpectrogramData(const std::vector<std::vector<float>>& data) {
    if (data.empty() || data[0].empty()) return data;
    
    std::vector<std::vector<float>> smoothed = data;
    const int timeSize = data.size();
    const int freqSize = data[0].size();
    
    // Kernel gaussiano simple 3x3
    const float kernel[3][3] = {
        {0.077847f, 0.123317f, 0.077847f},
        {0.123317f, 0.195346f, 0.123317f},
        {0.077847f, 0.123317f, 0.077847f}
    };
    
    for (int t = 1; t < timeSize - 1; ++t) {
        for (int f = 1; f < freqSize - 1; ++f) {
            float sum = 0.0f;
            for (int kt = -1; kt <= 1; ++kt) {
                for (int kf = -1; kf <= 1; ++kf) {
                    sum += data[t + kt][f + kf] * kernel[kt + 1][kf + 1];
                }
            }
            smoothed[t][f] = sum;
        }
    }
    
    return smoothed;
}

// Mapeo de color estilo espectrograma (azul -> verde -> amarillo -> rojo)
ImVec4 getSpectrogramColor(float intensity) {
    // Clamp intensity to [0, 1]
    intensity = std::max(0.0f, std::min(1.0f, intensity));
    
    // Color mapping: Dark Blue -> Cyan -> Green -> Yellow -> Red
    ImVec4 color;
    
    if (intensity < 0.25f) {
        // Dark Blue to Cyan
        float t = intensity / 0.25f;
        color = ImVec4(0.0f, t * 0.5f, 0.5f + t * 0.5f, 1.0f);
    }
    else if (intensity < 0.5f) {
        // Cyan to Green
        float t = (intensity - 0.25f) / 0.25f;
        color = ImVec4(0.0f, 0.5f + t * 0.5f, 1.0f - t * 0.5f, 1.0f);
    }
    else if (intensity < 0.75f) {
        // Green to Yellow
        float t = (intensity - 0.5f) / 0.25f;
        color = ImVec4(t, 1.0f, 0.5f - t * 0.5f, 1.0f);
    }
    else {
        // Yellow to Red
        float t = (intensity - 0.75f) / 0.25f;
        color = ImVec4(1.0f, 1.0f - t * 0.5f, 0.0f, 1.0f);
    }
    
    return color;
}

// Convertir coeficientes DWT a bandas de frecuencia de alta resolución
std::vector<float> computeHighResBands(const std::vector<float>& dwtCoeffs, int numBands) {
    std::vector<float> highResBands(numBands, 0.0f);
    if (dwtCoeffs.empty()) return highResBands;
    
    int dwtSize = std::min((int)dwtCoeffs.size(), DWT_COEFFICIENTS_TO_SHOW);
    
    // Mapear coeficientes DWT a bandas de alta resolución con interpolación
    for (int i = 0; i < numBands; ++i) {
        float dwtPos = (float)i / numBands * dwtSize;
        int idx1 = (int)dwtPos;
        int idx2 = std::min(idx1 + 1, dwtSize - 1);
        float frac = dwtPos - idx1;
        
        // Interpolación lineal entre coeficientes adyacentes
        float val1 = std::abs(dwtCoeffs[idx1]);
        float val2 = std::abs(dwtCoeffs[idx2]);
        highResBands[i] = val1 * (1.0f - frac) + val2 * frac;
    }
    
    return highResBands;
}

// -----------------------------------------------
// Decodifica un frame binario
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
// Clase SerialPort
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
// Implementación de Wavelet (Haar DWT)
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
                                g_dwtNeedsUpdate[i] = true;
                                
                                int counter = g_spectrogramUpdateCounter.fetch_add(1);
                                if (counter % SPECTROGRAM_UPDATE_INTERVAL == 0) {
                                    std::vector<float> recentData(g_channelData[i].begin(), g_channelData[i].end());
                                    
                                    std::vector<float> dwtCoeffs = compute_haart_dwt(recentData);
                                    std::vector<float> highResBands = computeHighResBands(dwtCoeffs, SPECTROGRAM_FREQ_BANDS);

                                    if (!highResBands.empty()) {
                                        g_bandEnergyHistory[i].push_back(highResBands);
                                        if (g_bandEnergyHistory[i].size() > ENERGY_HISTORY_SIZE) {
                                            g_bandEnergyHistory[i].pop_front();
                                        }
                                    }
                                    
                                    std::vector<float> bandEnergies = calculate_band_energy(dwtCoeffs);
                                    g_lastDwtCoeffs[i] = dwtCoeffs;
                                    g_lastBandEnergies[i] = bandEnergies;
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

    std::string portName;
#ifdef _WIN32
    portName = "COM5";
#else
    portName = "/dev/ttyUSB0";
#endif
    std::thread serialThread(readSpikeShield_thread, portName);

    auto lastFrameTime = std::chrono::high_resolution_clock::now();
    const double targetFrameTime = 1.0 / 30.0;

    while (!glfwWindowShouldClose(window) && g_keepRunning)
    {
        auto now = std::chrono::high_resolution_clock::now();
        double deltaTime = std::chrono::duration<double>(now - lastFrameTime).count();
        
        if (deltaTime < targetFrameTime) {
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>((targetFrameTime - deltaTime) * 1000)));
            continue;
        }
        lastFrameTime = now;

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            int width, height;
            glfwGetWindowSize(window, &width, &height);
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImVec2(width, height));
            
            ImGui::Begin("Visor de Canales DWT y Espectrograma", nullptr, 
                        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | 
                        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | 
                        ImGuiWindowFlags_NoScrollbar);

            ImGui::Text("Puerto: %s", portName.c_str());
            ImGui::SameLine();
            ImGui::Text(" | Fs: %.1f Hz", g_sampleFreq.load());
            ImGui::SameLine();
            ImGui::Text(" | Espectrograma: %d bandas de frecuencia", SPECTROGRAM_FREQ_BANDS);
            ImGui::Separator();

            ImGui::DragFloatRange2("Zoom Y (DWT Global)", &g_sharedDwtPlotRange.min, &g_sharedDwtPlotRange.max, 1.0f, -4096.0f, 4096.0f);
            ImGui::Separator();

            std::vector<std::vector<float>> plotData(NUM_CHANNELS);
            std::vector<std::vector<std::vector<float>>> energyHistory(NUM_CHANNELS);
            std::vector<bool> needsUpdate(NUM_CHANNELS);
            
            {
                std::lock_guard<std::mutex> lock(g_dataMutex);
                for (int i = 0; i < NUM_CHANNELS; ++i) {
                    plotData[i].assign(g_channelData[i].begin(), g_channelData[i].end());
                    energyHistory[i].assign(g_bandEnergyHistory[i].begin(), g_bandEnergyHistory[i].end());
                    needsUpdate[i] = g_dwtNeedsUpdate[i];
                    g_dwtNeedsUpdate[i] = false;
                }
            }
            
            float contentWidth = ImGui::GetContentRegionAvail().x;
            const float DWT_PLOT_HEIGHT = 120.0f; 
            const float SPACER_HEIGHT = 5.0f;
            
            float availableHeight = ImGui::GetContentRegionAvail().y;
            float requiredDwtHeight = (DWT_PLOT_HEIGHT + SPACER_HEIGHT) * NUM_CHANNELS;
            float spectrogramHeight = availableHeight - requiredDwtHeight - 5.0f;
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

                ImGui::Text("Canal %d - Coeficientes Haar DWT (N=%d)", i + 1, PLOT_BUFFER_SIZE);

                std::vector<float> waveletCoeffs;
                std::vector<float> bandEnergies;
                
                if (needsUpdate[i] || g_lastDwtCoeffs[i].empty()) {
                    waveletCoeffs = compute_haart_dwt(plotData[i]);
                    bandEnergies = calculate_band_energy(waveletCoeffs);
                    g_lastDwtCoeffs[i] = waveletCoeffs;
                    g_lastBandEnergies[i] = bandEnergies;
                } else {
                    waveletCoeffs = g_lastDwtCoeffs[i];
                    bandEnergies = g_lastBandEnergies[i];
                }
                
                if (!waveletCoeffs.empty()) {
                    ImVec4 bgColor = ImGui::GetStyleColorVec4(ImGuiCol_FrameBg);
                    int predominantBandIndex = -1;
                    
                    if (!bandEnergies.empty()) {
                        auto maxIt = std::max_element(bandEnergies.begin(), bandEnergies.end());
                        predominantBandIndex = std::distance(bandEnergies.begin(), maxIt);
                        
                        ImVec4 predominantColor = g_bands[predominantBandIndex].color;
                        bgColor = ImVec4(
                            std::min(bgColor.x + predominantColor.x * 0.15f, 1.0f),
                            std::min(bgColor.y + predominantColor.y * 0.15f, 1.0f),
                            std::min(bgColor.z + predominantColor.z * 0.15f, 1.0f),
                            1.0f
                        );
                    }

                    ImGui::BeginChild("DWT_Plot_Child", ImVec2(contentWidth, DWT_PLOT_HEIGHT), true, ImGuiWindowFlags_NoScrollbar);
                    
                    ImVec2 plot_min = ImGui::GetCursorScreenPos();
                    ImVec2 plot_max = ImVec2(plot_min.x + ImGui::GetContentRegionAvail().x, plot_min.y + ImGui::GetContentRegionAvail().y);
                    ImGui::GetWindowDrawList()->AddRectFilled(plot_min, plot_max, ImGui::GetColorU32(bgColor));

                    ImGui::Text("Banda Predominante: %s (RMS: %.2f)", 
                        (predominantBandIndex != -1) ? g_bands[predominantBandIndex].name.c_str() : "N/A",
                        (predominantBandIndex != -1) ? bandEnergies[predominantBandIndex] : 0.0f
                    );

                    ImGui::PlotLines("", waveletCoeffs.data(), DWT_COEFFICIENTS_TO_SHOW, 0, 
                                     nullptr, 
                                     g_sharedDwtPlotRange.min, g_sharedDwtPlotRange.max,
                                     ImVec2(ImGui::GetContentRegionAvail().x, DWT_PLOT_HEIGHT - ImGui::GetTextLineHeightWithSpacing() * 2), sizeof(float));
                    
                    ImGui::EndChild();
                }
                
                ImGui::Dummy(ImVec2(contentWidth, SPACER_HEIGHT));

                // NUEVO: Espectrograma mejorado tipo FL Studio (solo canal 1)
                if (i == 0) {
                    ImGui::Separator();
                    ImGui::Text("Canal %d: Espectrograma de Alta Resolución (Tiempo vs Frecuencia)", i + 1);
                    
                    if (energyHistory[i].size() > 5) {
                        ImGui::BeginChild("Spectrogram_Child", ImVec2(contentWidth, spectrogramHeight), true, ImGuiWindowFlags_NoScrollbar);
                        
                        float specWidth = ImGui::GetContentRegionAvail().x;
                        float specHeight = ImGui::GetContentRegionAvail().y;

                        ImVec2 spec_min = ImGui::GetCursorScreenPos();
                        
                        // Suavizar datos antes de renderizar
                        std::vector<std::vector<float>> smoothedData = smoothSpectrogramData(energyHistory[i]);
                        
                        // Normalización adaptativa
                        float maxEnergy = 1.0f;
                        for(const auto& frameEnergy : smoothedData) {
                            for (float energy : frameEnergy) {
                                maxEnergy = std::max(maxEnergy, energy);
                            }
                        }
                        
                        // Renderizar con interpolación bilineal
                        int timeSteps = smoothedData.size();
                        int freqBands = SPECTROGRAM_FREQ_BANDS;
                        
                        float pixelsPerTime = specWidth / timeSteps;
                        float pixelsPerFreq = specHeight / freqBands;
                        
                        // Dibujar píxeles interpolados (efecto de manchas suaves)
                        for (int px = 0; px < (int)specWidth; px += 2) {
                            for (int py = 0; py < (int)specHeight; py += 2) {
                                // Mapear posición de píxel a coordenadas de datos
                                float t = (float)px / specWidth * (timeSteps - 1);
                                float f = (float)py / specHeight * (freqBands - 1);
                                
                                int t1 = (int)t;
                                int t2 = std::min(t1 + 1, timeSteps - 1);
                                int f1 = (int)f;
                                int f2 = std::min(f1 + 1, freqBands - 1);
                                
                                float tx = t - t1;
                                float ty = f - f1;
                                
                                // Interpolación bilineal
                                float val = bilinearInterpolate(
                                    smoothedData[t1][f1],
                                    smoothedData[t1][f2],
                                    smoothedData[t2][f1],
                                    smoothedData[t2][f2],
                                    tx, ty
                                );
                                
                                // Normalizar y aplicar mapeo de color
                                float intensity = val / maxEnergy;
                                intensity = std::pow(intensity, 0.7f); // Ajuste gamma para mejor contraste
                                
                                ImVec4 color = getSpectrogramColor(intensity);
                                
                                // Dibujar rectángulo de 2x2 píxeles para suavidad
                                ImVec2 p1 = ImVec2(spec_min.x + px, spec_min.y + py);
                                ImVec2 p2 = ImVec2(spec_min.x + px + 2, spec_min.y + py + 2);
                                ImGui::GetWindowDrawList()->AddRectFilled(p1, p2, ImGui::GetColorU32(color));
                            }
                        }
                        
                        // Marcadores de bandas cerebrales (eje Y) - De arriba (Gamma) a abajo (Delta)
                        struct BandMarker {
                            const char* label;
                            float freqCenter; // Frecuencia central de la banda
                            ImVec4 color;
                        };
                        
                        const BandMarker bandMarkers[] = {
                            {"Gamma (30-60 Hz)", 45.0f, ImVec4(1.0f, 0.0f, 0.5f, 1.0f)},  // Magenta
                            {"Beta (13-30 Hz)", 21.5f, ImVec4(1.0f, 0.5f, 0.0f, 1.0f)},   // Naranja
                            {"Alpha (8-13 Hz)", 10.5f, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)},   // Amarillo
                            {"Theta (4-8 Hz)", 6.0f, ImVec4(0.0f, 1.0f, 0.0f, 1.0f)},     // Verde
                            {"Delta (<4 Hz)", 2.0f, ImVec4(0.0f, 0.5f, 1.0f, 1.0f)}       // Azul
                        };
                        
                        float maxFreq = 60.0f; // Frecuencia máxima
                        
                        for (int i = 0; i < 5; ++i) {
                            // Calcular posición Y basada en la frecuencia (invertida: arriba = alta frecuencia)
                            float normalizedFreq = bandMarkers[i].freqCenter / maxFreq;
                            float yPos = spec_min.y + (1.0f - normalizedFreq) * specHeight;
                            
                            ImVec2 labelPos = ImVec2(spec_min.x + 5, yPos - 8);
                            
                            // Dibujar etiqueta con color de la banda
                            ImGui::GetWindowDrawList()->AddText(labelPos, 
                                ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 1.0f, 0.9f)), 
                                bandMarkers[i].label);
                            
                            // Línea guía con color de la banda (más sutil)
                            ImVec2 lineStart = ImVec2(spec_min.x, yPos);
                            ImVec2 lineEnd = ImVec2(spec_min.x + specWidth, yPos);
                            ImVec4 lineColor = bandMarkers[i].color;
                            lineColor.w = 0.2f; // Transparencia
                            ImGui::GetWindowDrawList()->AddLine(lineStart, lineEnd, 
                                ImGui::GetColorU32(lineColor), 1.5f);
                        }
                        
                        // Etiqueta de tiempo
                        ImVec2 timeLabelPos = ImVec2(spec_min.x + specWidth - 120, spec_min.y + specHeight - 20);
                        ImGui::GetWindowDrawList()->AddText(timeLabelPos, 
                            ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 1.0f, 0.8f)), 
                            "Tiempo reciente ->");
                        
                        // Barra de escala de color (leyenda)
                        float legendX = spec_min.x + specWidth - 30;
                        float legendY = spec_min.y + 40;
                        float legendHeight = specHeight - 80;
                        
                        for (int i = 0; i < 50; ++i) {
                            float intensity = (float)i / 50.0f;
                            ImVec4 color = getSpectrogramColor(intensity);
                            ImVec2 p1 = ImVec2(legendX, legendY + (49 - i) * (legendHeight / 50));
                            ImVec2 p2 = ImVec2(legendX + 15, p1.y + (legendHeight / 50) + 1);
                            ImGui::GetWindowDrawList()->AddRectFilled(p1, p2, ImGui::GetColorU32(color));
                        }
                        
                        // Bordes de la leyenda
                        ImGui::GetWindowDrawList()->AddRect(
                            ImVec2(legendX, legendY), 
                            ImVec2(legendX + 15, legendY + legendHeight),
                            ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 1.0f, 0.5f))
                        );
                        
                        // Etiquetas de intensidad
                        ImGui::GetWindowDrawList()->AddText(
                            ImVec2(legendX - 35, legendY - 5), 
                            ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 1.0f, 0.8f)), 
                            "Max"
                        );
                        ImGui::GetWindowDrawList()->AddText(
                            ImVec2(legendX - 35, legendY + legendHeight - 10), 
                            ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 1.0f, 0.8f)), 
                            "Min"
                        );

                        ImGui::EndChild();
                    } else {
                        ImGui::Text("Acumulando datos para el espectrograma...");
                        ImGui::Dummy(ImVec2(contentWidth, spectrogramHeight));
                    }
                }
                
                ImGui::Separator();
                ImGui::PopID(); 
            }

            ImGui::End();
        }

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

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