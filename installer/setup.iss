[Setup]
AppName=Oxi Rust
AppVersion=0.1.0
DefaultDirName={pf}\OxiRust

[Files]
Source: "..\rust\target\release\oxi_rust.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Oxi Rust"; Filename: "{app}\oxi_rust.exe"
