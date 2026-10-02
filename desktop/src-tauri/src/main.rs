// Release builds must never open a console window on Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    stellaria_lib::run();
}
