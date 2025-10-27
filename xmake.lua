set_languages("c++23")

add_rules("mode.debug", "mode.release")
add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})

add_requires("imgui", { configs = { opengl3 = true, glfw = true }})
add_requires("glfw")
add_requires("glad")

target("eeg_app")
    set_kind("binary")
    add_includedirs("include")
    add_headerfiles("include/*.hpp")
    add_files("src/**.cpp")
    add_packages("glfw", "imgui", "glad")