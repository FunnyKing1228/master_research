# -*- mode: python ; coding: utf-8 -*-
"""
P302 Microgrid AI - PyInstaller spec
Package the GUI, control scripts, deployment model, and dependencies.
"""
import os

PROJECT_ROOT = os.getcwd()
DEPLOY_EXPERIMENT = 'v22_flow_power_limited_gpu300'
DEPLOY_MODEL_FILE = 'best_sac_model.pth'
MODEL_SOURCE = os.path.join(
    PROJECT_ROOT,
    'experiments',
    DEPLOY_EXPERIMENT,
    'models',
    DEPLOY_MODEL_FILE,
)
EXPERIMENT_CONFIG_SOURCE = os.path.join(
    PROJECT_ROOT,
    'experiments',
    DEPLOY_EXPERIMENT,
    'configs',
    'experiment_config.yaml',
)

if not os.path.exists(MODEL_SOURCE):
    raise FileNotFoundError(f"Deployment model not found: {MODEL_SOURCE}")
if not os.path.exists(EXPERIMENT_CONFIG_SOURCE):
    raise FileNotFoundError(f"Experiment config not found: {EXPERIMENT_CONFIG_SOURCE}")

a = Analysis(
    # Main GUI entry point.
    [os.path.join(PROJECT_ROOT, 'gui', 'ai_control_gui.py')],

    pathex=[PROJECT_ROOT],

    binaries=[],

    # Bundled data files: (source, destination in bundle).
    datas=[
        # Control scripts used by subprocess mode dispatch.
        (os.path.join(PROJECT_ROOT, 'control', '__init__.py'),            'control'),
        (os.path.join(PROJECT_ROOT, 'control', 'solar_test_collect.py'),  'control'),
        (os.path.join(PROJECT_ROOT, 'control', 'run_deployment.py'),      'control'),
        (os.path.join(PROJECT_ROOT, 'control', 'run_online_control.py'),  'control'),
        (os.path.join(PROJECT_ROOT, 'control', 'io_protocol.py'),         'control'),

        # Core modules imported by run_deployment.py.
        (os.path.join(PROJECT_ROOT, 'core', 'sac_agent.py'),       'core'),
        (os.path.join(PROJECT_ROOT, 'core', 'microgrid_env.py'),   'core'),
        (os.path.join(PROJECT_ROOT, 'core', 'safety_net.py'),      'core'),

        # SoH predictor
        (os.path.join(PROJECT_ROOT, 'core', 'soh_predictor'),      'core/soh_predictor'),

        # Deployment model: v22 flow-power-limited power+flow policy + deployment guards.
        (MODEL_SOURCE, 'models'),
        (EXPERIMENT_CONFIG_SOURCE, 'configs'),

        # Configuration files.
        (os.path.join(PROJECT_ROOT, 'configs', 'config_p302_sim.yaml'),  'configs'),

        # Load schedule.
        (os.path.join(PROJECT_ROOT, 'load_pattern.txt'),  '.'),
    ],

    hiddenimports=[
        'tkinter', 'tkinter.ttk', 'tkinter.messagebox',
        'tkinter.filedialog', 'tkinter.scrolledtext',
        'torch', 'numpy', 'yaml',
        'csv', 'json', 'argparse', 'sympy',
        # core modules that deployment script imports
        'core.sac_agent', 'core.microgrid_env', 'core.safety_net',
        'core.soh_predictor', 'core.soh_predictor.inference',
        # control modules（--mode dispatch 需要匯入）
        'control', 'control.io_protocol',
        'control.solar_test_collect', 'control.run_deployment',
    ],

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Plotting/scientific packages are not needed at deployment runtime.
        'matplotlib', 'scipy', 'pandas', 'PIL', 'Pillow',
        'plotly', 'bokeh', 'altair', 'seaborn', 'holoviews',
        'skimage', 'sklearn', 'scikit-learn', 'scikit-image',
        # Jupyter/IPython
        'IPython', 'notebook', 'jupyter', 'jupyter_core',
        'jupyter_client', 'jupyter_server', 'jupyterlab',
        'nbconvert', 'nbformat', 'ipykernel', 'ipywidgets',
        # Torch extensions.
        'torchaudio', 'torchvision', 'torchtext',
        # Qt is not needed; the GUI uses tkinter.
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'sip',
        # Documentation packages.
        'sphinx', 'sphinxcontrib', 'docutils', 'alabaster',
        # Other unused packages.
        'lxml', 'pygments', 'pytest', 'setuptools', '_pytest',
        'pyviz_comms', 'panel', 'param', 'intake', 'dask',
        'distributed', 'fsspec', 'cytoolz', 'toolz',
        'numba', 'llvmlite', 'cffi', 'cryptography', 'bcrypt',
        'zmq', 'tornado', 'jinja2', 'markupsafe',
        'chardet', 'charset_normalizer', 'certifi', 'urllib3',
        'requests', 'httpx', 'aiohttp',
        'h5py', 'tables', 'xlrd', 'openpyxl',
        'astropy', 'statsmodels',
        'conda', 'anaconda_navigator',
        'gymnasium', 'gym', 'pygame', 'box2d',
        # TensorFlow is not needed and makes the package very large.
        'tensorflow', 'tensorboard', 'keras',
        'google', 'google.protobuf', 'grpc', 'grpcio',
        'tensorboard_data_server', 'tensorboard_plugin_wit',
        'absl', 'astunparse', 'flatbuffers', 'gast',
        'opt_einsum', 'pasta', 'termcolor', 'wrapt',
        'tensorflow_estimator', 'tf_keras',
        # AWS/cloud packages are not needed.
        'boto3', 'botocore', 's3transfer',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='P302_AI_GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='P302_AI_GUI',
)
