  *	)\????t@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap,I???p??!1???#O@)@7n1??1??gȅ?H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::MapL?k?˴?!V_?/?}8@)z?c??T??1????0@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat<???r???!gg?A?? @)??7??d??1u???m@:Preprocessing2v
?Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[36]::Concatenates.?Ueߕ?!???f?@)T?d???1???9??@:Preprocessing2T
Iterator::Root::ParallelMapV2?% ??*??!y?]~7@)?% ??*??1y?]~7@:Preprocessing2E
Iterator::Root?}?[?~??!?.??w? @)毐?2???1,?^???
@:Preprocessing2v
?Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[37]::Concatenate{????!P?P!g@)?@,?9$??1?l?q??@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch)? ???!r??@??@))? ???1r??@??@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat,??26t??!???_0?@)?k??=}?1`???7@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?q?
??!R"?7??P@)?%jj?z?1S?О??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??1ZGUc?!???[????)??1ZGUc?1???[????:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?<?E~?`?!͊??d??)?<?E~?`?1͊??d??:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[36]::Concatenate[1]::FromTensor??	L?uK?!??/h+??)??	L?uK?1??/h+??:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[37]::Concatenate[1]::FromTensor??????I?!?	f??^??)??????I?1?	f??^??:Preprocessing2?
OIterator::Root::ParallelMapV2::Zip[0]::FlatMap[37]::Concatenate[0]::TensorSlice???E?!?-x????)???E?1?-x????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JCPU_ONLYb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.