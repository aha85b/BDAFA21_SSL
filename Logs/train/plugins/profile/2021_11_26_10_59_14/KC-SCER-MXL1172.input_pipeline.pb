  *	??????U@2v
?Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[18]::Concatenate:?6U???!??Wܪ?B@)??7?ܘ??1Tĥ??@@:Preprocessing2T
Iterator::Root::ParallelMapV2???f???!???4?5@)???f???1???4?5@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat%???7ی?!??kE=0@)????u??1"?qm?(@:Preprocessing2E
Iterator::Root?
?|$%??!.?8?G.@@)C???|͂?1??f???$@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipG6u??!i?c??P@)2t??ׁ?1?vx?#@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?o`r?Ȣ?!.S?D@)?ZӼ?m?1???A%@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorl??g?i?!n6?d4h@)l??g?i?1n6?d4h@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[18]::Concatenate[1]::FromTensor?iP4`a?!???J@)?iP4`a?1???J@:Preprocessing2?
OIterator::Root::ParallelMapV2::Zip[0]::FlatMap[18]::Concatenate[0]::TensorSlice??I???R?!?9ԑ???)??I???R?1?9ԑ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JCPU_ONLYb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.