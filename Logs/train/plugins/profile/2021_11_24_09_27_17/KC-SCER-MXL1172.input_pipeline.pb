  *	??"???r@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap
?8?*??!?z?v*?M@)??w??1?@?m)I@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map??˚X???!??{???9@)'???S??1???l?,@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat[??	m??!??ܞ?&@)?,
?(z??1?????L%@:Preprocessing2T
Iterator::Root::ParallelMapV2H?Ȱ?7??!3?Ȣd?@)H?Ȱ?7??13?Ȣd?@:Preprocessing2v
?Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[36]::Concatenate??)x
??!A??J/@)?A??????1\?????@:Preprocessing2E
Iterator::Root????I??!???:??"@)7?$??1??F???@:Preprocessing2v
?Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[37]::Concatenateعi3NC??!͇+?q1
@)?]=???1`??80@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatJ??{ds??!t&????@)?ŊLÀ?1??4?@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetchȷw??{?!>????@)ȷw??{?1>????@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip	?p???!N???5P@)N)???]r?11ا޽??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor!V?a?b?!e+??G=??)!V?a?b?1e+??G=??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range6??\^?!-??rR???)6??\^?1-??rR???:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[37]::Concatenate[1]::FromTensor8?*5{?E?!?}m?????)8?*5{?E?1?}m?????:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[36]::Concatenate[1]::FromTensoru?B?!f??5?l??)u?B?1f??5?l??:Preprocessing2?
OIterator::Root::ParallelMapV2::Zip[0]::FlatMap[37]::Concatenate[0]::TensorSliceŏ1w-!??!a9?'???)ŏ1w-!??1a9?'???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JCPU_ONLYb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.