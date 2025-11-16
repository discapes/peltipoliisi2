project(
  'peltipoliisi2',
  'cpp',
  version : '0.1',
  meson_version : '>= 1.3.0',
  default_options : ['warning_level=3', 'cpp_std=c++20'],
)

dependencies = [
	dependency('opencv4'),
  dependency('fftw3'),
  dependency('mlpack')
]

exe = executable(
  'peltipoliisi2',
  [
    'src/main.cpp',
    'src/event_reader.cpp',
    'src/rpm_estimator.cpp',
    'src/cluster_worker.cpp'
  ],
  install : true,
  dependencies : dependencies,
  cpp_args : ['-pedantic', '-Wall', '-Wextra']
)

test('basic', exe)
