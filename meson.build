project(
  'peltipoliisi2',
  'cpp',
  version : '0.1',
  meson_version : '>= 1.3.0',
  default_options : ['warning_level=3', 'cpp_std=c++20'],
)

opencv_dep = dependency('opencv4')
fftw_dep = dependency('fftw3')
mlpack_dep = dependency('mlpack')
torch_dep = dependency('Torch', method : 'cmake', required : true)

dependencies = [
  opencv_dep,
  fftw_dep,
  mlpack_dep,
  torch_dep
]

if torch_dep.found()
  add_project_arguments('-DTRAJECTORY_PREDICTOR_WITH_TORCH=1', language : 'cpp')
endif

exe = executable(
  'peltipoliisi2',
  [
    'src/main.cpp',
    'src/event_reader.cpp',
    'src/rpm_from_fft.cpp',
    'src/estimate_rpms.cpp',
    'src/cluster_worker.cpp',
    'src/render.cpp',
    'src/trajectory_predictor.cpp',
  ],
  install : true,
  dependencies : dependencies,
  cpp_args : ['-pedantic', '-Wall', '-Wextra']
)

test('basic', exe)
