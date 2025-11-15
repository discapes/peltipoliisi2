project(
  'peltipoliisi2',
  'cpp',
  version : '0.1',
  meson_version : '>= 1.3.0',
  default_options : ['warning_level=3', 'cpp_std=c++20'],
)

dependencies = [
	dependency('opencv4'),
  dependency('fftw3')
]

exe = executable(
  'peltipoliisi2',
  [
    'peltipoliisi2.cpp',
    'event_reader.cpp',
  ],
  install : true,
  dependencies : dependencies,
)

test('basic', exe)
