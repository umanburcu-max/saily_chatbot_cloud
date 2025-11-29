# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = [('templates', 'templates'), ('embeds', 'embeds')]
datas += collect_data_files('tiktoken')
datas += collect_data_files('tiktoken_ext.openai_public')


a = Analysis(
    ['app_embedding_crm_prompt.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['tiktoken_ext.openai_public', 'pypdf', 'pypdf.errors'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='app_embedding_crm_prompt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
