/*
 *
 *                   _/_/_/    _/_/   _/    _/ _/_/_/    _/_/
 *                  _/   _/ _/    _/ _/_/  _/ _/   _/ _/    _/
 *                 _/_/_/  _/_/_/_/ _/  _/_/ _/   _/ _/_/_/_/
 *                _/      _/    _/ _/    _/ _/   _/ _/    _/
 *               _/      _/    _/ _/    _/ _/_/_/  _/    _/
 *
 *             ***********************************************
 *                              PandA Project
 *                     URL: http://panda.dei.polimi.it
 *                       Politecnico di Milano - DEIB
 *                        System Architectures Group
 *             ***********************************************
 *                Copyright (C) 2025 Politecnico di Milano
 *
 *   This file is part of the PandA framework.
 *
 *   Licensed under the Apache License with BAMBU exceptions, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   A copy of the License can be found in the root directory of this repository.
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 */
/**
 * @file hls_stream.h
 * @brief Implementation of hls::stream object.
 *
 * @author Michele Fiorito <michele.fiorito@polimi.it>
 */
#ifndef __HLS_STREAM_H
#define __HLS_STREAM_H

#include "ac_channel.h"

namespace hls
{
   template <typename T>
   using stream = ac_channel<T>;

} // namespace hls
#endif
