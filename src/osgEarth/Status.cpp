/* -*-c++-*- */
/* osgEarth - Geospatial SDK for OpenSceneGraph
 * Copyright 2020 Pelican Mapping
 * http://osgearth.org
 *
 * osgEarth is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 */
#include <osgEarth/Status>
#include <osgEarth/StringUtils>

using namespace osgEarth;
using namespace osgEarth::Util;

#define LC "[Status] "

const osgEarth::Status osgEarth::STATUS_OK;

std::string osgEarth::Status::_codeText[6] = {
    "No error",
    "Resource unavailable",
    "Service unavailable",
    "Configuration error",
    "Assertion failure",
    "Error"
};
