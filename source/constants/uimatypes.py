# Experiment resources related to the MuLMS-AZ corpus (CODI 2023).
# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
This module contains UIMA CAS types used in the data reader.
"""

SENT_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
MATSCI_SENT_TYPE = "webanno.custom.MatSci_Sentence"
TOKEN_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
SKIPPING_TYPE = "webanno.custom.MatSci_Skipping"
ENTITY_TYPE = "webanno.custom.MatSci_Entity"
RELATION_TYPE = "webanno.custom.MatSci_Relations"
PASSAGE_TYPE = "webanno.custom.MatSci_Passage"
DOCUMENT_METADATA_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData"
